/**
 * Conduit Panel Module
 * Sidebar UI for managing workflows and settings
 */

import { app } from "/scripts/app.js";
import { toggleSocketTag, renameSocketTag, isTagNameTaken, getDefaultName } from './core.js';
import { collectTaggedSockets } from './tagged-sockets.js';
import { scheduleStyleUpdate } from './styles.js';
import { getWorkflowName } from './utils.js';
import { el } from './utils.js';
import { logMinimal, logNormal } from './logging.js';

export class ConduitPanel {
    constructor() {
        this.config = { output_folder: "", handler_command: "", always_run_handler: false };
        this.saveStatus = { saved: false, name: null, lastSaved: null };
        this.defaultHandlers = [];
        this.element = this.createPanel();
        this.loadConfig();
        this.checkSaveStatus();
    }

    createPanel() {
        const panel = el("div", {
            className: "conduit-panel",
            style: {
                display: "flex",
                flexDirection: "column",
                height: "100%",
                padding: "10px",
                boxSizing: "border-box",
                fontFamily: "Arial, sans-serif",
                fontSize: "13px",
                color: "#ccc",
                overflow: "hidden",
            }
        });

        // Add styles
        const style = document.createElement("style");
        style.textContent = `
            .conduit-panel h3 {
                margin: 0 0 10px 0;
                padding-bottom: 8px;
                border-bottom: 1px solid #444;
                font-size: 14px;
                color: #fff;
            }
            .conduit-panel .section {
                margin-bottom: 15px;
            }
            .conduit-panel input[type="text"] {
                width: 100%;
                padding: 6px 8px;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                color: #fff;
                font-size: 12px;
                box-sizing: border-box;
            }
            .conduit-panel input[type="text"]:focus {
                outline: none;
                border-color: #666;
            }
            .conduit-panel label {
                display: block;
                color: #aaa;
                margin-bottom: 4px;
                font-size: 11px;
            }
            .conduit-panel .btn {
                padding: 6px 12px;
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #fff;
                cursor: pointer;
                font-size: 12px;
            }
            .conduit-panel .btn:hover {
                background: #4a4a4a;
            }
            .conduit-panel .btn-primary {
                background: #4a7c59;
                border-color: #5a8c69;
            }
            .conduit-panel .btn-primary:hover {
                background: #5a8c69;
            }
            .conduit-panel .btn-danger {
                background: #7c4a4a;
                border-color: #8c5a5a;
            }
            .conduit-panel .btn-danger:hover {
                background: #8c5a5a;
            }
            .conduit-panel .workflow-header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }
            .conduit-panel .workflow-name-input {
                flex: 1;
                padding: 5px 8px;
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                color: #fff;
                font-size: 13px;
                font-weight: 500;
            }
            .conduit-panel .workflow-name-input:focus {
                outline: none;
                border-color: #666;
            }
            .conduit-panel .workflow-status {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 11px;
                color: #888;
                margin-bottom: 8px;
            }
            .conduit-panel .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }
            .conduit-panel .status-dot.saved {
                background: #5a8c69;
            }
            .conduit-panel .status-dot.unsaved {
                background: #8c8c5a;
            }
            .conduit-panel .status-dot.not-saved {
                background: #666;
            }
            .conduit-panel .workflow-actions {
                display: flex;
                gap: 8px;
                margin-bottom: 6px;
            }
            .conduit-panel .workflow-actions .btn {
                flex: 1;
            }
            .conduit-panel .workflow-stats {
                font-size: 11px;
                color: #666;
            }
            .conduit-panel .socket-item {
                display: flex;
                align-items: center;
                padding: 5px 8px;
                background: #2a2a2a;
                border-radius: 4px;
                margin-bottom: 3px;
                gap: 6px;
            }
            .conduit-panel .socket-item:hover {
                background: #333;
            }
            .conduit-panel .socket-side {
                font-size: 10px;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            .conduit-panel .socket-side.input {
                background: rgba(100, 150, 255, 0.3);
                color: #8af;
            }
            .conduit-panel .socket-side.output {
                background: rgba(255, 100, 100, 0.3);
                color: #f88;
            }
            .conduit-panel .socket-name {
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                cursor: pointer;
            }
            .conduit-panel .socket-name:hover {
                color: #fff;
            }
            .conduit-panel .icon-btn {
                background: none;
                border: none;
                color: #888;
                cursor: pointer;
                padding: 2px 4px;
                font-size: 14px;
            }
            .conduit-panel .icon-btn:hover {
                color: #fff;
            }
            .conduit-panel .checkbox-row {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 8px;
            }
            .conduit-panel .checkbox-row input {
                width: 14px;
                height: 14px;
            }
            .conduit-panel .scrollable {
                overflow-y: auto;
                flex: 1;
                min-height: 0;
            }
            .conduit-panel .empty-state {
                color: #666;
                font-style: italic;
                padding: 10px;
                text-align: center;
            }
        `;
        panel.appendChild(style);

        // Header
        panel.appendChild(el("h3", { textContent: "⚡ Conduit" }));

        // Settings Section
        const settingsSection = this.createSettingsSection();
        panel.appendChild(settingsSection);

        // Workflows Section
        const workflowsSection = this.createWorkflowsSection();
        panel.appendChild(workflowsSection);

        // Tagged Sockets Section (fills remaining space)
        const socketsSection = this.createSocketsSection();
        panel.appendChild(socketsSection);

        return panel;
    }

    createSettingsSection() {
        const section = el("div", { className: "section" });

        // Collapsible header
        const header = el("div", {
            style: {
                display: "flex",
                alignItems: "center",
                cursor: "pointer",
                marginBottom: "8px",
            },
            onclick: () => {
                const content = section.querySelector(".settings-content");
                const arrow = header.querySelector(".arrow");
                if (content.style.display === "none") {
                    content.style.display = "block";
                    arrow.textContent = "▼";
                } else {
                    content.style.display = "none";
                    arrow.textContent = "▶";
                }
            }
        }, [
            el("span", { className: "arrow", textContent: "▶", style: { marginRight: "6px", fontSize: "10px" } }),
            el("span", { textContent: "Settings", style: { fontWeight: "bold" } }),
        ]);
        section.appendChild(header);

        const content = el("div", {
            className: "settings-content",
            style: { display: "none", paddingLeft: "16px" }
        });

        // Output Folder
        content.appendChild(el("label", { textContent: "Output Folder" }));
        this.outputFolderInput = el("input", {
            style: { marginBottom: "8px" },
            oninput: () => this.saveConfigDebounced(),
        });
        this.outputFolderInput.type = "text";
        this.outputFolderInput.placeholder = "Leave empty for default";
        content.appendChild(this.outputFolderInput);

        // Handlers Section
        content.appendChild(el("label", { textContent: "Handlers", style: { marginTop: "8px" } }));
        this.handlersContainer = el("div", {
            style: {
                marginBottom: "8px",
                padding: "8px",
                background: "rgba(0,0,0,0.2)",
                borderRadius: "4px",
                maxHeight: "150px",
                overflowY: "auto",
            }
        });
        content.appendChild(this.handlersContainer);
        this.loadHandlers();  // Fetch and render handlers

        // Always Run Handler checkbox
        const checkboxRow = el("div", { className: "checkbox-row" });
        this.alwaysRunCheckbox = el("input");
        this.alwaysRunCheckbox.type = "checkbox";
        this.alwaysRunCheckbox.onchange = () => this.saveConfigDebounced();
        checkboxRow.appendChild(this.alwaysRunCheckbox);
        checkboxRow.appendChild(el("span", { textContent: "Always run handlers (even with no outputs)", style: { fontSize: "12px" } }));
        content.appendChild(checkboxRow);

        section.appendChild(content);
        return section;
    }

    createWorkflowsSection() {
        const section = el("div", { className: "section" });

        // Section title with folder button
        const header = el("div", {
            style: {
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "8px"
            }
        }, [
            el("span", { style: { fontWeight: "bold" }, textContent: "Workflow" }),
            el("button", {
                className: "icon-btn",
                innerHTML: "📁",
                style: { fontSize: "14px", padding: "2px 6px" },
                onclick: (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    this.openWorkflowsFolder();
                },
            }),
        ]);
        section.appendChild(header);

        // Current workflow name (read-only display)
        this.workflowNameDisplay = el("div", {
            style: {
                padding: "6px 10px",
                background: "rgba(0,0,0,0.2)",
                borderRadius: "4px",
                marginBottom: "8px",
                fontSize: "13px",
            },
            textContent: "No workflow loaded"
        });
        section.appendChild(this.workflowNameDisplay);

        // Hidden input to track name (for compatibility)
        this.workflowNameInput = el("input");
        this.workflowNameInput.type = "hidden";

        // Info text
        section.appendChild(el("div", {
            style: { fontSize: "11px", color: "#888", marginBottom: "8px" },
            textContent: "Conduit auto-saves when you save the workflow."
        }));

        // Status indicator
        this.statusRow = el("div", { className: "workflow-status" });
        this.statusDot = el("span", { className: "status-dot not-saved" });
        this.statusText = el("span", { textContent: "Not exported" });
        this.statusRow.appendChild(this.statusDot);
        this.statusRow.appendChild(this.statusText);
        section.appendChild(this.statusRow);

        // Action buttons - simplified
        const actions = el("div", { className: "workflow-actions" }, [
            el("button", {
                className: "btn",
                textContent: "Save Copy As...",
                onclick: () => this.saveWorkflowAs(),
            }),
        ]);
        section.appendChild(actions);

        // Stats line
        this.statsLine = el("div", { className: "workflow-stats" });
        section.appendChild(this.statsLine);

        return section;
    }

    createSocketsSection() {
        const section = el("div", {
            className: "section",
            style: {
                flex: "1",
                display: "flex",
                flexDirection: "column",
                minHeight: "0",
            }
        });

        const header = el("div", {
            style: {
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "8px",
            }
        }, [
            el("span", { textContent: "Tagged Sockets", style: { fontWeight: "bold" } }),
        ]);
        section.appendChild(header);

        // Socket list (scrollable)
        this.socketList = el("div", { className: "scrollable" });
        section.appendChild(this.socketList);

        return section;
    }

    // ---- Config Management ----

    async loadConfig() {
        try {
            const response = await fetch("/conduit/config");
            this.config = await response.json();
            this.outputFolderInput.value = this.config.output_folder || "";
            this.alwaysRunCheckbox.checked = this.config.always_run_handler || false;
            logNormal("config_loaded");
        } catch (e) {
            logNormal("config_load_error", { error: e.message });
        }
    }

    saveConfigDebounced() {
        if (this._saveTimeout) clearTimeout(this._saveTimeout);
        this._saveTimeout = setTimeout(() => this.saveConfig(), 500);
    }

    async saveConfig() {
        const config = {
            output_folder: this.outputFolderInput.value.trim(),
            always_run_handler: this.alwaysRunCheckbox.checked,
        };
        try {
            await fetch("/conduit/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config),
            });
            logNormal("config_saved");
        } catch (e) {
            logNormal("config_save_error", { error: e.message });
        }
    }

    // ---- Handler Management ----

    async loadHandlers() {
        try {
            const response = await fetch("/conduit/handlers");
            const data = await response.json();
            this.renderHandlers(data.handlers, data.defaults);
            logNormal("handlers_loaded", { count: Object.keys(data.handlers).length });
        } catch (e) {
            logNormal("handlers_load_error", { error: e.message });
            this.handlersContainer.innerHTML = '<div style="color: #888; font-size: 11px;">Failed to load handlers</div>';
        }
    }

    renderHandlers(handlers, defaults) {
        this.handlersContainer.innerHTML = "";

        // Store defaults for sparse storage comparison
        this.defaultHandlers = defaults || [];

        const handlerIds = Object.keys(handlers);
        if (handlerIds.length === 0) {
            this.handlersContainer.appendChild(el("div", {
                style: { color: "#888", fontSize: "11px" },
                textContent: "No handlers registered"
            }));
            return;
        }

        for (const handlerId of handlerIds) {
            const handler = handlers[handlerId];
            const isEnabled = defaults.includes(handlerId);

            const row = el("div", {
                className: "checkbox-row",
                style: { marginBottom: "4px" },
                "data-handler-id": handlerId,  // Store handler ID for getCurrentHandlerSelection
            });

            const checkbox = el("input");
            checkbox.type = "checkbox";
            checkbox.checked = isEnabled;
            checkbox.onchange = () => this.toggleHandler(handlerId, checkbox.checked);

            const label = el("div", {
                style: { flex: "1", marginLeft: "6px" }
            }, [
                el("div", {
                    style: { fontSize: "12px", fontWeight: "500" },
                    textContent: handler.name
                }),
                el("div", {
                    style: { fontSize: "10px", color: "#888" },
                    textContent: handler.description
                })
            ]);

            row.appendChild(checkbox);
            row.appendChild(label);
            this.handlersContainer.appendChild(row);
        }
    }

    async toggleHandler(handlerId, enabled) {
        const action = enabled ? "enable" : "disable";
        try {
            await fetch(`/conduit/handlers/${handlerId}/${action}`, { method: "POST" });
            logNormal("handler_toggled", { handlerId, action });
        } catch (e) {
            logNormal("handler_toggle_error", { handlerId, error: e.message });
            this.loadHandlers();
        }
    }

    // ---- Workflow Management ----

    /**
     * Open the conduit_workflows folder in the system file manager
     */
    async openWorkflowsFolder() {
        try {
            const response = await fetch("/conduit/workflows/open-folder", { method: "POST" });
            const result = await response.json();
            if (result.status !== "opened") {
                logNormal("folder_open_error", { message: result.message });
            }
        } catch (e) {
            logNormal("folder_open_error", { error: e.message });
        }
    }

    /**
     * Check if current workflow is saved to Conduit
     */
    async checkSaveStatus() {
        const currentName = getWorkflowName() || "";
        this.workflowNameInput.value = currentName;

        // Update the display
        if (currentName) {
            this.workflowNameDisplay.textContent = currentName;
        } else {
            this.workflowNameDisplay.textContent = "Unsaved workflow";
        }

        if (!currentName) {
            this.updateStatus("not-saved", "Save workflow to enable export");
            return;
        }

        try {
            const response = await fetch(`/conduit/workflows/${encodeURIComponent(currentName)}`);
            if (response.ok) {
                const data = await response.json();
                this.saveStatus = {
                    saved: true,
                    name: currentName,
                    lastSaved: data.updated_at || null,
                };
                this.updateStatus("saved", "Exported to Conduit");
            } else {
                // 404 is expected for workflows not yet exported - this is normal
                this.saveStatus = { saved: false, name: null, lastSaved: null };
                this.updateStatus("not-saved", "Not yet exported");
            }
        } catch (e) {
            this.saveStatus = { saved: false, name: null, lastSaved: null };
            this.updateStatus("not-saved", "Not yet exported");
        }

        this.updateStats();
    }

    /**
     * Update the status indicator
     */
    updateStatus(state, text) {
        this.statusDot.className = `status-dot ${state}`;
        this.statusText.textContent = text;
    }

    /**
     * Update the stats line showing tagged socket counts
     */
    updateStats() {
        const sockets = collectTaggedSockets();
        const inputs = sockets.filter(s => s.side === "input").length;
        const outputs = sockets.filter(s => s.side === "output").length;

        if (sockets.length === 0) {
            this.statsLine.textContent = "No sockets tagged";
        } else {
            this.statsLine.textContent = `${inputs} input${inputs !== 1 ? 's' : ''} · ${outputs} output${outputs !== 1 ? 's' : ''} tagged`;
        }
    }

    /**
     * Auto-save workflow to Conduit (called when ComfyUI saves)
     */
    async autoSaveWorkflow() {
        const name = getWorkflowName();
        if (!name) return;

        const sockets = collectTaggedSockets();
        if (sockets.length === 0) return;

        await this.doSave(name);
    }

    /**
     * Save workflow (used internally)
     */
    async saveWorkflow() {
        const name = getWorkflowName();
        if (!name) {
            app.extensionManager.toast.add({
                severity: "warn",
                summary: "Workflow not saved",
                detail: "Please save the workflow first (Ctrl+S)",
            });
            return;
        }

        const sockets = collectTaggedSockets();
        if (sockets.length === 0) {
            app.extensionManager.toast.add({
                severity: "info",
                summary: "No sockets tagged",
                detail: "Right-click sockets to tag them for export",
            });
            return;
        }

        await this.doSave(name);
    }

    /**
     * Save workflow with a new name (prompts user)
     */
    async saveWorkflowAs() {
        const currentName = this.workflowNameInput.value.trim() || getWorkflowName() || "";
        const newName = await app.extensionManager.dialog.prompt({
            title: "Save Conduit Workflow As",
            message: "Enter workflow name:",
            defaultValue: currentName,
        });

        if (!newName || !newName.trim()) return;

        const sockets = collectTaggedSockets();
        if (sockets.length === 0) {
            app.extensionManager.toast.add({
                severity: "info",
                summary: "No sockets tagged",
                detail: "Right-click sockets to tag them for export",
            });
            return;
        }

        this.workflowNameInput.value = newName.trim();
        await this.doSave(newName.trim());
    }

    /**
     * Get current handler selection from checkboxes.
     * Returns array of enabled handler IDs.
     */
    getCurrentHandlerSelection() {
        const selected = [];
        const checkboxes = this.handlersContainer?.querySelectorAll('input[type="checkbox"]');
        if (!checkboxes) return null;

        // Handler IDs are stored in a data attribute on the parent row
        checkboxes.forEach((checkbox, index) => {
            if (checkbox.checked) {
                // Get handler ID from the checkbox's context
                const row = checkbox.closest('.checkbox-row');
                if (row && row.dataset.handlerId) {
                    selected.push(row.dataset.handlerId);
                }
            }
        });

        return selected;
    }

    /**
     * Perform the actual save operation.
     * Implements sparse handler storage - only includes handlers if different from defaults.
     */
    async doSave(name) {
        // Include disabled sockets in save so API can filter by active status
        const sockets = collectTaggedSockets({ includeDisabled: true });
        const { output } = await app.graphToPrompt();

        // Build save payload
        const payload = {
            name: name,
            workflow: output,
            sockets: sockets,
        };

        // Sparse handler storage: only include if different from defaults
        const currentHandlers = this.getCurrentHandlerSelection();
        if (currentHandlers !== null && this.defaultHandlers) {
            // Sort both arrays to compare
            const currentSorted = [...currentHandlers].sort();
            const defaultSorted = [...this.defaultHandlers].sort();

            // Check if different from defaults
            const isDifferent = currentSorted.length !== defaultSorted.length ||
                currentSorted.some((h, i) => h !== defaultSorted[i]);

            if (isDifferent) {
                payload.handlers = currentHandlers;
            }
        }

        try {
            const response = await fetch("/conduit/workflows", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const result = await response.json();
            if (result.status === "saved") {
                logMinimal("workflow_saved", { name: result.name, sockets: sockets.length, isUpdate: result.is_update });
                this.saveStatus = {
                    saved: true,
                    name: name,
                    lastSaved: new Date().toISOString(),
                };
                this.updateStatus("saved", result.is_update ? "Updated" : "Saved to Conduit");
                this.updateStats();
            } else {
                logMinimal("workflow_save_failed", { name, message: result.message });
                this.updateStatus("not-saved", "Save failed");
            }
        } catch (e) {
            logMinimal("workflow_save_error", { name, error: e.message });
            this.updateStatus("not-saved", "Save failed");
        }
    }

    // ---- Tagged Sockets ----

    updateTaggedSockets() {
        const sockets = collectTaggedSockets();
        this.socketList.innerHTML = "";

        // Update stats line
        this.updateStats();

        if (sockets.length === 0) {
            this.socketList.appendChild(el("div", {
                className: "empty-state",
                textContent: "Right-click sockets to tag them",
            }));
            return;
        }

        // Sort: inputs first, then outputs
        sockets.sort((a, b) => {
            if (a.side !== b.side) return a.side === "input" ? -1 : 1;
            return a.tagName.localeCompare(b.tagName);
        });

        for (const socket of sockets) {
            const item = el("div", { className: "socket-item" }, [
                el("span", {
                    className: `socket-side ${socket.side}`,
                    textContent: socket.side === "input" ? "IN" : "OUT",
                }),
                el("span", {
                    className: "socket-name",
                    textContent: socket.tagName,
                    onclick: () => this.renameSocket(socket),
                }),
                el("span", {
                    style: { color: "#666", fontSize: "10px" },
                    textContent: socket.dataType,
                }),
                el("button", {
                    className: "icon-btn",
                    innerHTML: "✕",
                    onclick: () => this.removeSocket(socket),
                }),
            ]);
            this.socketList.appendChild(item);
        }
    }

    async renameSocket(socket) {
        const node = app.graph?.getNodeById(socket.nodeId);
        if (!node) return;

        const defaultName = getDefaultName(node, socket.side, socket.slotIndex);
        let currentName = socket.tagName;
        let errorMessage = null;

        while (true) {
            const message = errorMessage
                ? `⚠️ ${errorMessage}\n\nEnter new tag name:`
                : "Enter new tag name:";

            const newName = await app.extensionManager.dialog.prompt({
                title: "Rename Socket Tag",
                message: message,
                defaultValue: currentName,
            });

            // User cancelled
            if (newName === null) break;

            const trimmedName = newName.trim() || defaultName;

            // No change
            if (trimmedName === socket.tagName) break;

            // Check if name is taken by another socket
            if (isTagNameTaken(app.graph, trimmedName, socket.nodeId, socket.side, socket.slotIndex)) {
                errorMessage = `"${trimmedName}" is already used by another socket`;
                currentName = trimmedName;
                continue;
            }

            // Valid name - apply it
            renameSocketTag(node, socket.side, socket.slotIndex, trimmedName);
            this.updateTaggedSockets();
            break;
        }
    }

    removeSocket(socket) {
        const node = app.graph?.getNodeById(socket.nodeId);
        if (!node) return;

        // Use toggleSocketTag to properly handle custom names (marks inactive instead of deleting)
        toggleSocketTag(node, socket.side, socket.slotIndex);
        scheduleStyleUpdate();
        this.updateTaggedSockets();
    }

    // Called when panel becomes visible
    update() {
        this.updateTaggedSockets();
        this.checkSaveStatus();
    }
}

/**
 * Register the Conduit sidebar tab
 */
export function registerSidebarTab() {
    const panel = new ConduitPanel();
    window.conduitPanel = panel;

    app.extensionManager.registerSidebarTab({
        id: "conduit",
        icon: "pi pi-bolt",
        title: "Conduit",
        tooltip: "Conduit - Socket Tagging & Workflows",
        type: "custom",
        render: (container) => {
            // Clear container and re-add our panel to ensure clean state
            container.innerHTML = "";
            container.appendChild(panel.element);
            panel.update();
        },
    });

    logNormal("sidebar_registered");
}
