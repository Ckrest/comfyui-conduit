/**
 * Conduit Queue Hook Module
 * Queue interception and output injection
 */

import { app } from "/scripts/app.js";
import { collectTaggedOutputs, collectTaggedInputs, injectConduitOutputs } from './tagged-sockets.js';
import { getWorkflowName, generatePromptId } from './utils.js';
import { logMinimal, logNormal } from './logging.js';

/**
 * Track pending prompt IDs that need flushing
 * Maps ComfyUI's prompt_id -> our internal promptId
 */
const pendingPrompts = new Map();

/**
 * Flush outputs for a completed prompt - calls the handler
 */
async function flushPromptOutputs(promptId) {
    try {
        const response = await fetch(`/conduit/flush/${promptId}`, { method: "POST" });
        const result = await response.json();
        if (result.output_count > 0) {
            logMinimal("outputs_flushed", { promptId, count: result.output_count });
        }
    } catch (e) {
        logMinimal("flush_error", { promptId, error: e.message });
    }
}

/**
 * Register execution context with the backend before queuing.
 */
async function registerExecutionContext(promptId, context) {
    try {
        await fetch(`/conduit/context/${promptId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(context),
        });
    } catch (e) {
        logNormal("context_register_error", { promptId, error: e.message });
    }
}

/**
 * Update execution context (e.g., add comfy_prompt_id after queuing).
 */
async function updateExecutionContext(promptId, updates) {
    try {
        await fetch(`/conduit/context/${promptId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updates),
        });
    } catch (e) {
        logNormal("context_update_error", { promptId, error: e.message });
    }
}

/**
 * Hook into ComfyUI's queue system to inject our nodes.
 * Called from setup() hook - API is guaranteed to be ready.
 */
export function setupQueueHook() {
    const api = app.api;
    const originalQueuePrompt = api.queuePrompt.bind(api);

    api.queuePrompt = async function(number, { output, workflow }) {
        // Check if we have any tagged outputs
        const taggedOutputs = collectTaggedOutputs();

        if (taggedOutputs.length > 0) {
            const promptId = generatePromptId();

            // Also collect tagged inputs for context
            const taggedInputs = collectTaggedInputs();
            const workflowName = getWorkflowName();

            // Register execution context BEFORE queuing
            await registerExecutionContext(promptId, {
                source: "frontend",
                workflow_name: workflowName,
                tagged_inputs: taggedInputs,
                tagged_outputs: taggedOutputs,
                // Frontend doesn't process inputs, so these are empty
                inputs_applied: {},
                inputs_defaulted: [],
                warnings: [],
            });

            // Clone the output to avoid modifying the original
            const modifiedOutput = JSON.parse(JSON.stringify(output));

            // Inject our capture nodes
            const injected = injectConduitOutputs(modifiedOutput, promptId);
            logNormal("queue_intercepted", { promptId, outputs: injected });

            // Call original with modified output
            const result = await originalQueuePrompt(number, { output: modifiedOutput, workflow });

            // Map ComfyUI's prompt_id to our internal promptId for flushing later
            if (result && result.prompt_id) {
                pendingPrompts.set(result.prompt_id, promptId);
                // Update context with ComfyUI's prompt_id
                updateExecutionContext(promptId, { comfy_prompt_id: result.prompt_id });
            }

            return result;
        }

        // No tagged outputs, proceed normally
        return originalQueuePrompt(number, { output, workflow });
    };

    // Listen for execution_success - fires for EACH completed prompt
    api.addEventListener("execution_success", (event) => {
        const comfyPromptId = event.detail?.prompt_id;
        if (comfyPromptId && pendingPrompts.has(comfyPromptId)) {
            const ourPromptId = pendingPrompts.get(comfyPromptId);
            logMinimal("execution_success", { promptId: ourPromptId });
            flushPromptOutputs(ourPromptId);
            pendingPrompts.delete(comfyPromptId);
        }
    });

    // Handle execution_error to clean up pending prompts
    api.addEventListener("execution_error", (event) => {
        const comfyPromptId = event.detail?.prompt_id;
        if (comfyPromptId && pendingPrompts.has(comfyPromptId)) {
            const ourPromptId = pendingPrompts.get(comfyPromptId);
            logMinimal("execution_error", { promptId: ourPromptId, error: event.detail?.exception_message });
            pendingPrompts.delete(comfyPromptId);
        }
    });

    logNormal("queue_hook_installed");
}
