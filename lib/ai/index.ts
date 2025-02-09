import { openai } from "@ai-sdk/openai";
import {
  wrapLanguageModel,
  customProvider,
  type LanguageModelV1,
  type LanguageModelV1CallOptions,
  type LanguageModelV1StreamPart,
} from "ai";
import { openrouter } from "@openrouter/ai-sdk-provider";
import { togetherai } from "@ai-sdk/togetherai";
import { deepseek } from "@ai-sdk/deepseek";
import { OpenAICompatibleChatLanguageModel } from "@ai-sdk/openai-compatible";
import {
  generateId,
  loadApiKey,
  withoutTrailingSlash,
} from "@ai-sdk/provider-utils";
import { z } from "zod";

import { customMiddleware } from "./custom-middleware";

// Default model to use if none specified
const DEFAULT_MODEL = "gpt-3.5-turbo";

// Error schema for LiteLLM responses
const liteLLMErrorSchema = z.object({
  error: z.object({
    message: z.string(),
    type: z.string().nullable().optional(),
    param: z.any().nullable().optional(),
    code: z.union([z.string(), z.number()]).nullable().optional(),
  }),
});

// LiteLLM provider settings interface
interface LiteLLMProviderSettings {
  baseURL?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  modelId?: string;
}

// Create LiteLLM provider instance with model ID
const createLiteLLMProvider = (options: LiteLLMProviderSettings = {}) => {
  const baseURL = withoutTrailingSlash(
    options.baseURL ?? process.env.LITELLM_API_URL ?? ""
  );
  const apiKey = options.apiKey ?? process.env.LITELLM_API_KEY;

  return customProvider({
    languageModels: {
      default: new OpenAICompatibleChatLanguageModel(
        options.modelId || DEFAULT_MODEL,
        {
          // Add any LiteLLM specific settings here
        },
        {
          provider: "litellm.chat",
          url: ({ path }) => `${baseURL}${path}`,
          headers: () => ({
            Authorization: `Bearer ${loadApiKey({
              apiKey,
              environmentVariableName: "LITELLM_API_KEY",
              description: "LiteLLM API Key",
            })}`,
            ...options.headers,
          }),
          errorStructure: {
            errorSchema: liteLLMErrorSchema,
            errorToMessage: (error) => error.error.message,
            isRetryable: (response) =>
              response.status === 429 ||
              (response.status >= 500 && response.status <= 599),
          },
          supportsStructuredOutputs: true,
          defaultObjectGenerationMode: "json",
        }
      ),
    },
  });
};

// Type definition for valid reasoning models used for research and structured outputs
type ReasoningModel = (typeof VALID_REASONING_MODELS)[number];

// Valid reasoning models that can be used for research analysis and structured outputs
const VALID_REASONING_MODELS = [
  "o1",
  "o1-mini",
  "o3-mini",
  "deepseek-ai/DeepSeek-R1",
  "deepseek-reasoner",
  "gpt-4o",
  "deepseek-r1:32b",
] as const;

// Models that support JSON structured output
const JSON_SUPPORTED_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o3-mini",
  "deepseek-r1:32b",
] as const;

// Helper to check if model supports JSON
export const supportsJsonOutput = (modelId: string) =>
  JSON_SUPPORTED_MODELS.includes(
    modelId as (typeof JSON_SUPPORTED_MODELS)[number]
  );

// Get reasoning model from env, with JSON support info
const REASONING_MODEL = process.env.REASONING_MODEL || "o1-mini";
const BYPASS_JSON_VALIDATION = process.env.BYPASS_JSON_VALIDATION === "true";

// Helper to get the reasoning model based on user's selected model
function getReasoningModel(modelId: string) {
  // If already using a valid reasoning model, keep using it
  if (VALID_REASONING_MODELS.includes(modelId as ReasoningModel)) {
    return modelId;
  }

  const configuredModel = REASONING_MODEL;

  if (!VALID_REASONING_MODELS.includes(configuredModel as ReasoningModel)) {
    const fallback = "o1-mini";
    console.warn(
      `Invalid REASONING_MODEL "${configuredModel}", falling back to ${fallback}`
    );
    return fallback;
  }

  // Warn if trying to use JSON with unsupported model
  if (!BYPASS_JSON_VALIDATION && !supportsJsonOutput(configuredModel)) {
    console.warn(
      `Warning: Model ${configuredModel} does not support JSON schema. Set BYPASS_JSON_VALIDATION=true to override`
    );
  }

  return configuredModel;
}

export const customModel = (
  apiIdentifier: string,
  forReasoning: boolean = false
) => {
  // Check which API services are available
  const hasLiteLLMServer =
    process.env.LITELLM_API_URL && process.env.LITELLM_API_URL !== "";
  const hasOpenRouterKey =
    process.env.OPENROUTER_API_KEY && process.env.OPENROUTER_API_KEY !== "****";

  // If it's for reasoning, get the appropriate reasoning model
  const modelId = forReasoning
    ? getReasoningModel(apiIdentifier)
    : apiIdentifier.replace(/-\d{4}-\d{2}-\d{2}$/, ""); // Remove date suffix

  // Try LiteLLM first if available
  if (hasLiteLLMServer) {
    const liteLLMProvider = createLiteLLMProvider({
      baseURL: process.env.LITELLM_API_URL,
      apiKey: process.env.LITELLM_API_KEY,
      modelId,
    });
    return wrapLanguageModel({
      model: liteLLMProvider.languageModel("default"),
      middleware: customMiddleware,
    });
  }

  // Fall back to OpenRouter if available
  if (hasOpenRouterKey) {
    return wrapLanguageModel({
      model: openrouter(modelId),
      middleware: customMiddleware,
    });
  }

  // Fall back to other providers based on model
  const model =
    modelId === "deepseek-ai/DeepSeek-R1"
      ? togetherai(modelId)
      : modelId === "deepseek-reasoner"
      ? deepseek(modelId)
      : openai(modelId);

  return wrapLanguageModel({
    model,
    middleware: customMiddleware,
  });
};
