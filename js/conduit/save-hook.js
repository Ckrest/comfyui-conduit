/**
 * Conduit Save Hook Module
 * Hook into ComfyUI's save commands to export Conduit workflow
 */

import { app } from "/scripts/app.js";
import { collectTaggedSockets } from './tagged-sockets.js';
import { getWorkflowName } from './utils.js';
import { logMinimal, logNormal } from './logging.js';

let _isExporting = false;

async function exportToConduit(workflowName) {
    if (_isExporting || !workflowName) return;

    // Include disabled sockets - they need to be saved with active: false
    const sockets = collectTaggedSockets({ includeDisabled: true });
    if (sockets.length === 0) return;

    _isExporting = true;
    try {
        await window.conduitPanel?.doSave(workflowName);
    } catch (error) {
        logMinimal("save_hook_error", { workflowName, error: error.message });
    } finally {
        _isExporting = false;
    }
}

/**
 * Hook into ComfyUI's command system to export Conduit workflow on save.
 */
export function setupSaveHook() {
    const commands = app.extensionManager?.command?.commands;
    if (!commands) {
        logNormal("save_hook_skipped", { reason: "command_system_unavailable" });
        return;
    }

    const saveCommandIds = ['Comfy.SaveWorkflow', 'Comfy.SaveWorkflowAs'];
    let hooked = 0;

    for (const cmd of commands) {
        if (saveCommandIds.includes(cmd.id)) {
            const originalFn = cmd.function;

            cmd.function = async function(...args) {
                await originalFn.apply(this, args);
                const workflowName = getWorkflowName();
                if (workflowName) {
                    exportToConduit(workflowName);
                }
            };

            hooked++;
        }
    }

    if (hooked > 0) {
        logNormal("save_hook_installed", { commands: hooked });
    }
}
