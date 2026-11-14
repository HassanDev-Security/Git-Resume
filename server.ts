import express, { Request, Response } from "express";
import path from "path";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// USER CONFIGURATION HARDCODED FALLBACKS
// You can replace these empty strings directly with your secret keys 
// so the application remains fully functional and autonomous when downloaded!
const HARDCODED_GEMINI_API_KEY = "AIzaSyC7aw8ZDemtSCQFm5PB9z5tabXRmdWv-84"; // 👈 Enter your Gemini API Key here (e.g. "AIzaSy...")
const HARDCODED_GITHUB_TOKEN = "";     // 👈 Enter your GitHub Personal Access Token here (e.g. "ghp_...")

const app = express();
const PORT = 3000;

// Solid Cross-Origin Resource Sharing (CORS) Middleware to support custom domains like https://gitpro-resume.hkdeveloper.tech
app.use((req, res, next) => {
  const allowedOrigins = [
    "https://gitpro-resume.hkdeveloper.tech",
    "http://localhost:3000"
  ];
  const origin = req.headers.origin;
  if (origin && allowedOrigins.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Access-Control-Allow-Credentials", "true");
  } else {
    res.setHeader("Access-Control-Allow-Origin", "https://gitpro-resume.hkdeveloper.tech");
    res.setHeader("Access-Control-Allow-Credentials", "true");
  }
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  
  if (req.method === "OPTIONS") {
    res.sendStatus(200);
    return;
  }
  next();
});

app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ limit: "15mb", extended: true }));

// Initialize Gemini Client Lazily/Safely
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY || HARDCODED_GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY is missing. Please set it in process.env.GEMINI_API_KEY or configure HARDCODED_GEMINI_API_KEY in server.ts");
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

/**
 * Clean markdown or conversational text wrapping around JSON structures to ensure smooth and steady parses.
 */
function cleanAndParseJSON(rawText: string): any {
  let cleaned = rawText.trim();
  if (cleaned.startsWith("```json")) {
    cleaned = cleaned.slice(7);
  } else if (cleaned.startsWith("```")) {
    cleaned = cleaned.slice(3);
  }
  if (cleaned.endsWith("```")) {
    cleaned = cleaned.slice(0, -3);
  }
  cleaned = cleaned.trim();
  
  try {
    return JSON.parse(cleaned);
  } catch (err) {
    const firstCurly = cleaned.indexOf("{");
    const lastCurly = cleaned.lastIndexOf("}");
    if (firstCurly !== -1 && lastCurly !== -1 && lastCurly > firstCurly) {
      const extracted = cleaned.substring(firstCurly, lastCurly + 1);
      return JSON.parse(extracted);
    }
    throw err;
  }
}

/**
 * Executes a Gemini API call with exponential backoff retries and auto-fallback to alternative models
 * if the primary model is throttled or experiencing high density load (e.g. HTTP 503/429/UNAVAILABLE).
 * Features solid exponential backoff with jitter and dynamic configuration relaxed fallbacks.
 */
async function generateContentWithRetry(
  ai: GoogleGenAI,
  params: { contents: any; config: any },
  models: string[] = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]
): Promise<any> {
  let lastError: any = null;

  for (const modelName of models) {
    const isPrimary = modelName === models[0];
    const maxAttempts = isPrimary ? 2 : 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        console.log(`[Gemini API] Requesting portfolio synthesis. Model: ${modelName} (Attempt ${attempt}/${maxAttempts})`);
        
        // Prepare a model-specific configuration draft
        const activeConfig = { ...params.config };
        
        // Only relax responseSchema as an absolute last resort if we are on the final attempt of the final model
        const isAbsoluteLastResort = modelName === models[models.length - 1] && attempt === maxAttempts;
        if (isAbsoluteLastResort) {
          if (activeConfig.responseSchema) {
            console.log(`[Gemini API] Relaxing responseSchema parameter on final fallback model to maximize compatibility...`);
            delete activeConfig.responseSchema;
            activeConfig.systemInstruction = `${activeConfig.systemInstruction || ""}\n\nIMPORTANT: You must return the output as a valid raw JSON object. Do not wrap in markdown or backticks unless needed. Ensure it matches key properties: {"summary": string, "top_skills": string[], "featured_projects": [{"name": string, "description": string, "tech_stack": string[], "metrics": string}]}`;
          }
        }

        const response = await ai.models.generateContent({
          model: modelName,
          contents: params.contents,
          config: activeConfig,
        });
        
        console.log(`[Gemini API] Success using model: ${modelName}`);
        return response;
      } catch (error: any) {
        lastError = error;
        const errMsg = error.message || error.toString() || "";
        console.warn(`[Gemini API] Error on model ${modelName} (Attempt ${attempt}/${maxAttempts}): ${errMsg}`);

        // Define if error matches structural network retry potentials (503 Service Unavailable or 429 Rate Exceeded)
        const isRetryable =
          errMsg.includes("503") ||
          errMsg.includes("429") ||
          errMsg.includes("UNAVAILABLE") ||
          errMsg.includes("ResourceExhausted") ||
          errMsg.includes("demand") ||
          errMsg.includes("temporary") ||
          errMsg.includes("busy") ||
          errMsg.includes("overloaded");

        if (isRetryable && attempt < maxAttempts) {
          // Responsive fast retry delay with a tiny bit of jitter to avoid blocking request pipelines
          const jitterDelay = 1200 + Math.floor(Math.random() * 800);
          console.log(`[Gemini API] Retrying model ${modelName} in ${jitterDelay}ms due to temporary error...`);
          await new Promise((resolve) => setTimeout(resolve, jitterDelay));
        } else {
          // Break inner loop immediately to transition to the next fallback model lane
          break;
        }
      }
    }
    console.log(`[Gemini API] Model ${modelName} was unsuccessful or unavailable. Transitioning to next fallback model...`);
  }

  throw lastError || new Error("Failed to generate content after all resilient model fallbacks and retries were exhausted.");
}

// REST route: Fetch GitHub details and repositories safely (preventing IP block or exposing client credentials)
app.get("/api/github/user/:username", async (req: Request, res: Response) => {
  const { username } = req.params;

  if (!username) {
    res.status(400).json({ error: "Username is required" });
    return;
  }

  // Set up Authorization headers if GITHUB_TOKEN is specified in environment
  const headers: Record<string, string> = {
    "User-Agent": "github-portfolio-analyzer-app",
    Accept: "application/vnd.github.v3+json",
  };

  const activeGithubToken = process.env.GITHUB_TOKEN || HARDCODED_GITHUB_TOKEN;
  if (activeGithubToken) {
    headers["Authorization"] = `token ${activeGithubToken}`;
  }

  try {
    // 1. Fetch user profile data
    const profileRes = await fetch(`https://api.github.com/users/${username}`, { headers });
    if (!profileRes.ok) {
      if (profileRes.status === 404) {
        res.status(404).json({ error: "GitHub user not found." });
        return;
      }
      const errText = await profileRes.text();
      res.status(profileRes.status).json({ error: `GitHub API error (Profile): ${errText || profileRes.statusText}` });
      return;
    }
    const profileData = await profileRes.json();

    // 2. Fetch user's active repositories (last 30)
    const reposRes = await fetch(`https://api.github.com/users/${username}/repos?sort=updated&per_page=30`, { headers });
    if (!reposRes.ok) {
      const errText = await reposRes.text();
      res.status(reposRes.status).json({ error: `GitHub API error (Repos): ${errText || reposRes.statusText}` });
      return;
    }
    const reposData = await reposRes.json();

    res.json({
      profile: profileData,
      repos: reposData,
    });
  } catch (error: any) {
    console.error("Error fetching GitHub data:", error);
    res.status(500).json({ error: error.message || "Failed to fetch GitHub data." });
  }
});

// REST route: Secure backend route to generate resume using Gemini
app.post("/api/generate-resume", async (req: Request, res: Response) => {
  const { profile, repos } = req.body;

  if (!profile || !repos) {
    res.status(400).json({ error: "Profile and repositories data are required in request body." });
    return;
  }

  try {
    const ai = getGeminiClient();

    // Extract repository facts to pass to the model
    const processedRepos = repos
      .map((r: any) => ({
        name: r.name,
        description: r.description || "No description provided",
        language: r.language || "Not specified",
        stars: r.stargazers_count,
        forks: r.forks_count,
        updated_at: r.updated_at,
      }))
      // Sort to grab most active or highly-starred repositories
      .sort((a: any, b: any) => b.stars + b.forks - (a.stars + a.forks))
      .slice(0, 10); // Pass a concise summary of the top 10 repos to keep context clean and accurate

    const portfolioContext = {
      user: {
        name: profile.name || profile.login,
        login: profile.login,
        bio: profile.bio || "Full-Stack Developer",
        location: profile.location || "Remote",
        company: profile.company || "Independent",
        blog: profile.blog || "",
        followers: profile.followers,
      },
      repositories: processedRepos,
    };

    // Strict system prompt matching user spec
    const systemPrompt = `You are an expert technical recruiter. Analyze the following raw GitHub profile and repository data. Generate a highly professional developer resume.
Return ONLY a valid JSON object matching the requested schema. Make the resume read with great authority, impact-driven summaries, and accurate technology mapping. Do not use generic filler words. Ensure skills align perfectly with the languages used in their top repositories.`;

    const promptText = `Please analyze this developer portfolio data and compile a professional resume:
${JSON.stringify(portfolioContext, null, 2)}`;

    // Call Gemini API with strong structured schema via resilient retry-and-fallback mechanism
    const response = await generateContentWithRetry(ai, {
      contents: promptText,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: "A 3-sentence professional recruiter-style summary of the developer based on their skills and activity.",
            },
            top_skills: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "The top 8 technical skills, frameworks, or languages (e.g., 'TypeScript', 'Node.js', 'React') extracted from their repos.",
            },
            featured_projects: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING, description: "Name of the repository." },
                  description: { type: Type.STRING, description: "A high-impact resume description beginning with an action verb, outlining what the project does and its technical challenges." },
                  tech_stack: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Array of frameworks, libraries, or tools used." },
                  metrics: { type: Type.STRING, description: "Professional metrics summary, e.g. '53 stars with active contributions' or similar high-value statement." },
                },
                required: ["name", "description", "tech_stack", "metrics"],
              },
              description: "List of the top 3-4 most substantive, production-grade projects based on stars and details.",
            },
          },
          required: ["summary", "top_skills", "featured_projects"],
        },
      },
    });

    const parsedText = response.text;
    if (!parsedText) {
      res.status(500).json({ error: "Gemini did not return any readable portfolio resume content." });
      return;
    }

    // Parse output using robust fallback extractor to ensure valid JSON payload structure
    const resumePayload = cleanAndParseJSON(parsedText);
    res.json(resumePayload);
  } catch (error: any) {
    console.error("Error generating resume with Gemini API:", error);
    res.status(500).json({ error: error.message || "Failed to analyze portfolio using Gemini API." });
  }
});

// Vite Integration
async function start() {
  if (process.env.NODE_ENV !== "production") {
    // Dynamically import Vite only in development to prevent overhead in production bundles
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production delivery: serve compiled SPA
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server starting on http://localhost:${PORT}`);
  });
}

start();
