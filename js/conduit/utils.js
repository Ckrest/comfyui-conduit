/**
 * Conduit Utils Module
 * Shared utility functions
 */

import { app } from "/scripts/app.js";

/**
 * Create a styled DOM element with optional children.
 * Provides a simple way to create elements without JSX or templates.
 *
 * @param {string} tag - HTML tag name
 * @param {object} options - Element options (className, style, textContent, etc.)
 * @param {Array} children - Child elements or strings
 * @returns {HTMLElement}
 */
export function el(tag, options = {}, children = []) {
    const element = document.createElement(tag);

    if (options.className) {
        element.className = options.className;
    }
    if (options.style) {
        Object.assign(element.style, options.style);
    }
    if (options.textContent) {
        element.textContent = options.textContent;
    }
    if (options.innerHTML) {
        element.innerHTML = options.innerHTML;
    }
    if (options.onclick) {
        element.addEventListener('click', options.onclick);
    }
    if (options.onchange) {
        element.onchange = options.onchange;
    }
    if (options.oninput) {
        element.oninput = options.oninput;
    }
    for (const [key, value] of Object.entries(options)) {
        if (key.startsWith('data-')) {
            element.setAttribute(key, value);
        }
    }

    for (const child of children) {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else if (child) {
            element.appendChild(child);
        }
    }

    return element;
}

/**
 * Get the current workflow name.
 * Returns null if the workflow hasn't been saved yet.
 */
export function getWorkflowName() {
    return app.extensionManager?.workflow?.activeWorkflow?.filename ?? null;
}

/**
 * Generate a unique prompt ID for this execution.
 */
export function generatePromptId() {
    const now = new Date();
    const timestamp = now.toISOString().replace(/[-:T]/g, '').slice(0, 14);
    const random = Math.random().toString(36).slice(2, 6);
    return `${timestamp}_${random}`;
}
