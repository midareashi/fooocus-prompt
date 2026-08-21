// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

function gradioButton(selector) {
    const element = gradioApp().querySelector(selector);
    if (!element) return null;
    if (element.matches('button')) return element;
    return element.querySelector('button');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function getPersonLikenessPathsInput() {
    return gradioApp().querySelector(
        '#person_likeness_paths textarea, #person_likeness_paths input, textarea#person_likeness_paths, input#person_likeness_paths'
    );
}

function setPersonLikenessPaths(paths) {
    const input = getPersonLikenessPathsInput();
    if (!input) return;
    const value = JSON.stringify(paths);
    const valueSetter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), 'value')?.set;
    if (valueSetter) {
        valueSetter.call(input, value);
    } else {
        input.value = value;
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
}

function refreshPersonLikenessGallery() {
    setTimeout(function() {
        gradioApp().querySelector('#person_likeness_refresh_button')?.click();
    }, 20);
}

function installPersonLikenessRemoveButtons() {
    const input = getPersonLikenessPathsInput();
    const gallery = gradioApp().querySelector('#person_likeness_gallery');
    if (!input || !gallery) return;

    let paths = [];
    try {
        paths = JSON.parse(input.value || '[]');
    } catch {
        paths = [];
    }

    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item, index) {
        const path = paths[index] || '';
        item.dataset.personLikenessPath = path;

        let button = item.querySelector('.person-likeness-remove');
        if (!button) {
            button = document.createElement('button');
            button.type = 'button';
            button.className = 'person-likeness-remove';
            button.setAttribute('aria-label', 'Remove photo');
            button.textContent = 'X';
            item.appendChild(button);
        }

        button.dataset.personLikenessPath = path;
        button.type = 'button';
        button.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            const removePath = button.dataset.personLikenessPath || item.dataset.personLikenessPath || '';
            let currentPaths = [];
            try {
                currentPaths = JSON.parse(input.value || '[]');
            } catch {
                currentPaths = [];
            }
            const nextPaths = currentPaths.filter(function(path, pathIndex) {
                return removePath ? path !== removePath : pathIndex !== index;
            });
            setPersonLikenessPaths(nextPaths);
            item.remove();
            refreshPersonLikenessGallery();
        };
    });
}

function getGenerationApplyIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_apply_index textarea, #selected_generation_apply_index input, textarea#selected_generation_apply_index, input#selected_generation_apply_index'
    );
}

function getGenerationRemoveIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_remove_index textarea, #selected_generation_remove_index input, textarea#selected_generation_remove_index, input#selected_generation_remove_index'
    );
}

function getGenerationDeleteIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_delete_index textarea, #selected_generation_delete_index input, textarea#selected_generation_delete_index, input#selected_generation_delete_index'
    );
}

function getGenerationQualityIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_quality_index textarea, #selected_generation_quality_index input, textarea#selected_generation_quality_index, input#selected_generation_quality_index'
    );
}

function getGenerationFavoriteIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_favorite_index textarea, #selected_generation_favorite_index input, textarea#selected_generation_favorite_index, input#selected_generation_favorite_index'
    );
}

function getGenerationDetailIndexInput() {
    return gradioApp().querySelector(
        '#selected_generation_detail_index textarea, #selected_generation_detail_index input, textarea#selected_generation_detail_index, input#selected_generation_detail_index'
    );
}

function getGenerationFavoriteButton() {
    return gradioButton('#favorite_selected_generation_button');
}

function getGenerationDetailButton() {
    return gradioButton('#show_selected_generation_detail_button');
}

function getQuickPreviewGenerationIndicesInput() {
    return gradioApp().querySelector(
        '#quick_preview_generation_indices textarea, #quick_preview_generation_indices input, textarea#quick_preview_generation_indices, input#quick_preview_generation_indices'
    );
}

function setHiddenActionValue(input, value) {
    if (!input) return false;
    const stringValue = String(value);
    const valueSetter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), 'value')?.set;
    if (valueSetter) {
        valueSetter.call(input, stringValue);
    } else {
        input.value = stringValue;
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
}

function setGenerationActionIndex(input, index) {
    return setHiddenActionValue(input, index);
}

function setGenerationApplyIndex(index) {
    return setGenerationActionIndex(getGenerationApplyIndexInput(), index);
}

function setGenerationRemoveIndex(index) {
    return setGenerationActionIndex(getGenerationRemoveIndexInput(), index);
}

function setGenerationDeleteIndex(index) {
    return setGenerationActionIndex(getGenerationDeleteIndexInput(), index);
}

function setGenerationQualityIndex(index) {
    return setGenerationActionIndex(getGenerationQualityIndexInput(), index);
}

function setGenerationFavoriteIndex(index) {
    return setGenerationActionIndex(getGenerationFavoriteIndexInput(), index);
}

function setGenerationDetailIndex(index) {
    return setGenerationActionIndex(getGenerationDetailIndexInput(), index);
}

function getQueueRemoveIdInput() {
    return gradioApp().querySelector(
        '#selected_queue_remove_id textarea, #selected_queue_remove_id input, textarea#selected_queue_remove_id, input#selected_queue_remove_id'
    );
}

function setQueueRemoveId(queueId) {
    return setHiddenActionValue(getQueueRemoveIdInput(), queueId);
}

function ensureGenerationHistoryButton(item, className, label, title, onClick) {
    let button = item.querySelector(`.${className}`);
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = `generation-history-action ${className}`;
        button.textContent = label;
        item.appendChild(button);
    }
    setAttributeIfChanged(button, 'aria-label', title);
    setAttributeIfChanged(button, 'title', title);
    button.onclick = onClick;
}

function setAttributeIfChanged(element, name, value) {
    if (element && element.getAttribute(name) !== String(value)) {
        element.setAttribute(name, value);
    }
}

function setFavoriteStarState(button, isFavorite) {
    if (!button) return;
    button.classList.toggle('history-favorite-active', isFavorite);
    const nextText = isFavorite ? '\u2605' : '\u2606';
    const nextLabel = isFavorite ? 'Remove favorite' : 'Add favorite';
    if (button.textContent !== nextText) {
        button.textContent = nextText;
    }
    setAttributeIfChanged(button, 'aria-label', nextLabel);
    setAttributeIfChanged(button, 'title', nextLabel);
}

function installGenerationHistoryApplyButtons() {
    const gallery = gradioApp().querySelector('#final_gallery');
    const applyButton = gradioButton('#apply_selected_image_config_button');
    const removeButton = gradioButton('#remove_selected_image_button');
    const deleteButton = gradioButton('#delete_selected_image_button');
    const qualityButton = gradioButton('#regenerate_selected_quality_button');
    const favoriteButton = getGenerationFavoriteButton();
    const detailButton = getGenerationDetailButton();
    const previewIndicesInput = getQuickPreviewGenerationIndicesInput();
    if (!gallery || !applyButton || !removeButton || !deleteButton || !qualityButton || !favoriteButton || !detailButton) return;

    let previewIndices = [];
    try {
        previewIndices = JSON.parse(previewIndicesInput?.value || '[]');
    } catch {
        previewIndices = [];
    }
    const previewIndexSet = new Set(previewIndices.map((index) => Number(index)));

    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item, index) {
        item.dataset.generationHistoryIndex = String(index);
        if (item.dataset.generationDetailsBound !== 'true') {
            item.addEventListener('click', function(event) {
                if (event.target.closest('.generation-history-action')) return;
                const detailIndex = item.dataset.generationHistoryIndex;
                if (setGenerationDetailIndex(detailIndex)) {
                    detailButton.click();
                }
            });
            item.dataset.generationDetailsBound = 'true';
        }

        const isTinyThumbnail = item.getBoundingClientRect().width < 80;
        item.classList.toggle('generation-config-apply-hidden', isTinyThumbnail);
        if (isTinyThumbnail) {
            item.querySelectorAll('.generation-history-action').forEach(function(button) {
                button.remove();
            });
            return;
        }

        ensureGenerationHistoryButton(item, 'generation-config-apply', 'Apply', 'Apply generation config', function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (setGenerationApplyIndex(index)) {
                applyButton.click();
            }
        });
        ensureGenerationHistoryButton(item, 'generation-history-remove', 'Remove', 'Remove from history', function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (setGenerationRemoveIndex(index)) {
                removeButton.click();
            }
        });
        ensureGenerationHistoryButton(item, 'generation-history-delete', 'Delete', 'Delete file and remove from history', function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (setGenerationDeleteIndex(index)) {
                deleteButton.click();
            }
        });
        ensureGenerationHistoryButton(item, 'generation-history-favorite', '\u2605', 'Favorite image', function(event) {
            event.preventDefault();
            event.stopPropagation();
            const starButton = event.currentTarget;
            if (setGenerationFavoriteIndex(index)) {
                starButton.classList.toggle('history-favorite-active');
                setFavoriteStarState(starButton, starButton.classList.contains('history-favorite-active'));
                favoriteButton.click();
            }
        });
        setFavoriteStarState(item.querySelector('.generation-history-favorite'),
            item.querySelector('.generation-history-favorite')?.classList.contains('history-favorite-active'));

        const existingQualityButton = item.querySelector('.generation-quality-regenerate');
        if (!previewIndexSet.has(index)) {
            existingQualityButton?.remove();
            return;
        }

        ensureGenerationHistoryButton(item, 'generation-quality-regenerate', 'Quality 60', 'Regenerate quick preview at Quality, 60 steps', function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (setGenerationQualityIndex(index)) {
                qualityButton.click();
            }
        });
    });
}

function getHistorySelectionModeInput() {
    return gradioApp().querySelector('#history_selection_mode textarea, #history_selection_mode input');
}

function isHistoryDebugEnabled() {
    try {
        const params = new URLSearchParams(window.location.search);
        if (params.get('history_debug') === '1') {
            return true;
        }
        return window.localStorage?.getItem('fooocus_history_debug') === '1';
    } catch {
        return false;
    }
}

function getHistoryDaySelectionModeInput() {
    return gradioApp().querySelector('#history_day_selection_mode textarea, #history_day_selection_mode input');
}

function getHistorySelectedIdsInput() {
    return gradioApp().querySelector('#history_selected_image_ids_json textarea, #history_selected_image_ids_json input');
}

function getHistorySelectThumbnailIdInput() {
    return gradioApp().querySelector('#history_select_thumbnail_image_id textarea, #history_select_thumbnail_image_id input');
}

function getHistorySelectThumbnailButton() {
    return gradioButton('#history_select_thumbnail_button');
}

function getHistoryRemoveSelectedIdInput() {
    return gradioApp().querySelector('#history_remove_selected_image_id textarea, #history_remove_selected_image_id input');
}

function getHistoryRemoveSelectedButton() {
    return gradioButton('#history_remove_selected_image_button');
}

function getHistoryDeleteSelectedIdInput() {
    return gradioApp().querySelector('#history_delete_selected_image_id textarea, #history_delete_selected_image_id input');
}

function getHistoryDeleteSelectedButton() {
    return gradioButton('#history_delete_selected_image_button');
}

function getHistoryApplySelectedIdInput() {
    return gradioApp().querySelector('#history_apply_selected_image_id textarea, #history_apply_selected_image_id input');
}

function getHistoryApplySelectedButton() {
    return gradioButton('#history_apply_selected_image_button');
}

function getHistoryQualitySelectedIdInput() {
    return gradioApp().querySelector('#history_quality_selected_image_id textarea, #history_quality_selected_image_id input');
}

function getHistoryQualitySelectedButton() {
    return gradioButton('#history_quality_selected_image_button');
}

function getHistoryToggleFavoriteIdInput() {
    return gradioApp().querySelector('#history_toggle_favorite_image_id textarea, #history_toggle_favorite_image_id input');
}

function getHistoryToggleFavoriteButton() {
    return gradioButton('#history_toggle_favorite_button');
}

function getHistoryHideThumbnailIdInput() {
    return gradioApp().querySelector('#history_hide_thumbnail_image_id textarea, #history_hide_thumbnail_image_id input');
}

function getHistoryHideThumbnailButton() {
    return gradioButton('#history_hide_thumbnail_button');
}

function getHistoryConfigActionIdInput() {
    return gradioApp().querySelector('#history_config_action_image_id textarea, #history_config_action_image_id input');
}

function getHistoryLoadFullButton() {
    return gradioButton('#history_load_full_button');
}

function getHistoryReplacePromptButton() {
    return gradioButton('#history_replace_prompt_button');
}

function getHistoryAppendPromptButton() {
    return gradioButton('#history_append_prompt_button');
}

function getHistorySendToInpaintButton() {
    return gradioButton('#history_send_to_inpaint_button');
}

function setHistorySelectionMode(event) {
    const input = getHistorySelectionModeInput();
    let mode = 'single';
    if (event.shiftKey) {
        mode = 'shift';
    } else if (event.ctrlKey || event.metaKey) {
        mode = 'ctrl';
    }
    setHiddenActionValue(input, mode);
}

function setHistoryDaySelectionMode(event) {
    const input = getHistoryDaySelectionModeInput();
    let mode = 'single';
    if (event.shiftKey) {
        mode = 'shift';
    } else if (event.ctrlKey || event.metaKey) {
        mode = 'ctrl';
    }
    setHiddenActionValue(input, mode);
}

function getHistorySelectedIds() {
    const input = getHistorySelectedIdsInput();
    if (!input) return new Set();
    try {
        return new Set(JSON.parse(input.value || '[]').map((value) => Number(value)));
    } catch {
        return new Set();
    }
}

function getHistoryImageIdFromThumb(item) {
    const caption = item.querySelector('.caption, .thumbnail-label, p, span')?.textContent || item.textContent || '';
    const match = caption.match(/#(\d+)/);
    return match ? Number(match[1]) : null;
}

function getHistorySelectionRefFromThumb(item) {
    const caption = item.querySelector('.caption, .thumbnail-label, p, span')?.textContent || item.textContent || '';
    const stackMatch = caption.match(/stack:(\d+)/i);
    if (stackMatch) return `stack:${stackMatch[1]}`;
    const imageId = getHistoryImageIdFromThumb(item);
    return imageId === null ? '' : String(imageId);
}

function historyItemCaptionText(item) {
    return item.querySelector('.caption, .thumbnail-label, p, span')?.textContent || item.textContent || '';
}

function historyItemIsFavorite(item) {
    return /\bfav\b/i.test(historyItemCaptionText(item));
}

function historyItemIsHidden(item) {
    return /\bhidden\b/i.test(historyItemCaptionText(item));
}

function historyItemIsPreview(item) {
    return /\bpreview\b/i.test(historyItemCaptionText(item));
}

function triggerHistoryFavoriteToggle(imageId) {
    const input = getHistoryToggleFavoriteIdInput();
    const button = getHistoryToggleFavoriteButton();
    if (!input || !button || imageId === null) return false;
    if (!setHiddenActionValue(input, imageId)) return false;
    button.click();
    return true;
}

function triggerHistoryImageAction(input, button, imageId) {
    if (!input || !button || imageId === null) return false;
    if (!setHiddenActionValue(input, imageId)) return false;
    button.click();
    return true;
}

function ensureHistoryThumbnailBulkButton(container, className, label, title, actionButton) {
    let button = container.querySelector(`.${className}`);
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = `history-thumbnail-bulk-action ${className}`;
        button.textContent = label;
        container.appendChild(button);
    }
    setAttributeIfChanged(button, 'aria-label', title);
    setAttributeIfChanged(button, 'title', title);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        actionButton?.click();
    };
}

function installHistoryThumbnailBulkActions() {
    const gallery = gradioApp().querySelector('#history_thumbnail_gallery');
    const deleteButton = gradioButton('#history_bulk_delete_button');
    const favoriteButton = gradioButton('#history_bulk_favorite_button');
    const hideButton = gradioButton('#history_bulk_hide_button');
    if (!gallery || !deleteButton || !favoriteButton || !hideButton) return;

    const thumbnails = gallery.querySelector('.thumbnails');
    const actionHost = thumbnails?.parentElement || gallery;
    let actions = gallery.querySelector('.history-thumbnail-bulk-actions');
    if (!actions) {
        actions = document.createElement('div');
        actions.className = 'history-thumbnail-bulk-actions';
        actionHost.insertBefore(actions, thumbnails || null);
    } else if (actions.parentElement !== actionHost || actions.nextElementSibling !== thumbnails) {
        actionHost.insertBefore(actions, thumbnails || null);
    }
    ensureHistoryThumbnailBulkButton(actions, 'history-thumbnail-bulk-delete', '\u{1F5D1}',
        'Delete selected thumbnails and files', deleteButton);
    ensureHistoryThumbnailBulkButton(actions, 'history-thumbnail-bulk-favorite', '\u2605',
        'Favorite selected thumbnails', favoriteButton);
    ensureHistoryThumbnailBulkButton(actions, 'history-thumbnail-bulk-hide', '\u{1F441}',
        'Hide selected thumbnails', hideButton);
}

function ensureHistoryFavoriteButton(item, imageId) {
    if (imageId === null) {
        item.querySelector('.history-favorite-toggle')?.remove();
        item.querySelector('.history-hide-thumbnail')?.remove();
        return;
    }
    let button = item.querySelector('.history-favorite-toggle');
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'history-favorite-toggle';
        button.textContent = '\u2606';
        item.appendChild(button);
    }
    const isFavorite = historyItemIsFavorite(item);
    button.classList.toggle('history-favorite-active', isFavorite);
    setFavoriteStarState(button, isFavorite);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        if (triggerHistoryFavoriteToggle(imageId)) {
            const nextFavorite = !button.classList.contains('history-favorite-active');
            button.classList.toggle('history-favorite-active', nextFavorite);
            setFavoriteStarState(button, nextFavorite);
        }
    };
}

function ensureHistoryHideThumbnailButton(item, imageId, className = 'history-hide-thumbnail') {
    if (imageId === null) {
        item.querySelector(`.${className}`)?.remove();
        return;
    }
    let button = item.querySelector(`.${className}`);
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = className;
        button.textContent = '\u{1F441}';
        item.appendChild(button);
    }
    const title = historyItemIsHidden(item) ? 'Show in gallery' : 'Hide from gallery';
    setAttributeIfChanged(button, 'aria-label', title);
    setAttributeIfChanged(button, 'title', title);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        triggerHistoryImageAction(getHistoryHideThumbnailIdInput(), getHistoryHideThumbnailButton(), imageId);
    };
}

function triggerHistoryConfigAction(imageId, buttonGetter) {
    const input = getHistoryConfigActionIdInput();
    const button = buttonGetter();
    return triggerHistoryImageAction(input, button, imageId);
}

function installHistoryThumbnailSelection() {
    const gallery = gradioApp().querySelector('#history_thumbnail_gallery');
    if (!gallery) return;
    const selectedIds = getHistorySelectedIds();
    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item, index) {
        if (item.dataset.historySelectionInstalled !== 'true') {
            item.dataset.historySelectionInstalled = 'true';
            item.addEventListener('pointerdown', function(event) {
                setHistorySelectionMode(event);
                if (isHistoryDebugEnabled()) {
                    const debugImageRef = getHistorySelectionRefFromThumb(item);
                    console.debug('[HistoryDebug] pointerdown',
                        { hasShift: event.shiftKey, hasCtrl: event.ctrlKey || event.metaKey, ref: debugImageRef });
                }
            }, true);
            item.addEventListener('click', function(event) {
                if (event.target.closest('.history-favorite-toggle, .history-hide-thumbnail')) return;
                setHistorySelectionMode(event);
                const input = getHistorySelectThumbnailIdInput();
                const button = getHistorySelectThumbnailButton();
                const currentItems = Array.from(gallery.querySelectorAll('.thumbnail-item'));
                const currentIndex = currentItems.indexOf(item);
                if (currentIndex < 0) {
                    return;
                }
                const rawSelectionRef = getHistorySelectionRefFromThumb(item);
                const imageRef = `index:${currentIndex}` + (rawSelectionRef ? `|${rawSelectionRef}` : '');
                if (isHistoryDebugEnabled()) {
                    console.debug('[HistoryDebug] click payload', {
                        currentIndex,
                        rawSelectionRef,
                        imageRef,
                        target: event.target && event.target.className,
                        shift: event.shiftKey,
                        ctrl: event.ctrlKey || event.metaKey
                    });
                }
                if (!input || !button) return;
                if (setHiddenActionValue(input, imageRef)) {
                    if (isHistoryDebugEnabled()) {
                        console.debug('[HistoryDebug] triggering history select button', imageRef);
                    }
                    button.click();
                }
            }, true);
        }
        const imageId = getHistoryImageIdFromThumb(item);
        ensureHistoryFavoriteButton(item, imageId);
        ensureHistoryHideThumbnailButton(item, imageId);
        item.classList.toggle('history-thumbnail-selected', imageId !== null && selectedIds.has(imageId));
    });
}

function ensureHistorySelectedActionButton(item, className, label, title, imageId, inputGetter, buttonGetter) {
    let button = item.querySelector(`.${className}`);
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = `history-selected-action ${className}`;
        button.textContent = label;
        item.appendChild(button);
    }
    setAttributeIfChanged(button, 'aria-label', title);
    setAttributeIfChanged(button, 'title', title);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        triggerHistoryImageAction(inputGetter(), buttonGetter(), imageId);
    };
}

function closeHistorySelectedMenus(root) {
    const scope = root || gradioApp();
    scope.querySelectorAll('.history-selected-menu-open').forEach(function(item) {
        item.classList.remove('history-selected-menu-open');
    });
}

function ensureHistorySelectedConfigMenu(item, imageId) {
    let button = item.querySelector('.history-selected-kebab');
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'history-selected-action history-selected-kebab';
        button.textContent = '\u22EE';
        item.appendChild(button);
    }
    setAttributeIfChanged(button, 'aria-label', 'Image config actions');
    setAttributeIfChanged(button, 'title', 'Image config actions');

    let menu = item.querySelector('.history-selected-config-menu');
    if (!menu) {
        menu = document.createElement('div');
        menu.className = 'history-selected-config-menu';
        item.appendChild(menu);
    }

    const actions = [
        ['Load Full Config', getHistoryLoadFullButton],
        ['Replace Prompt', getHistoryReplacePromptButton],
        ['Append Prompt', getHistoryAppendPromptButton],
        ['Send to Inpaint', getHistorySendToInpaintButton]
    ];
    actions.forEach(function(action) {
        let menuButton = menu.querySelector(`button[data-history-config-action="${action[0]}"]`);
        if (!menuButton) {
            menuButton = document.createElement('button');
            menuButton.type = 'button';
            menuButton.dataset.historyConfigAction = action[0];
            menuButton.textContent = action[0];
            menu.appendChild(menuButton);
        }
        menuButton.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            item.classList.remove('history-selected-menu-open');
            triggerHistoryConfigAction(imageId, action[1]);
        };
    });

    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        const wasOpen = item.classList.contains('history-selected-menu-open');
        closeHistorySelectedMenus();
        item.classList.toggle('history-selected-menu-open', !wasOpen);
    };
}

function installHistorySelectedRemoveButtons() {
    const gallery = gradioApp().querySelector('#history_selected_gallery');
    if (!gallery) return;
    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item) {
        item.classList.add('history-selected-image-item');
        const imageId = getHistoryImageIdFromThumb(item);
        ensureHistorySelectedConfigMenu(item, imageId);
        ensureHistoryFavoriteButton(item, imageId);
        ensureHistorySelectedActionButton(item, 'history-selected-delete', '\u{1F5D1}', 'Delete image and file', imageId,
            getHistoryDeleteSelectedIdInput, getHistoryDeleteSelectedButton);
        ensureHistorySelectedActionButton(item, 'history-selected-remove', '\u2212', 'Remove from selected images', imageId,
            getHistoryRemoveSelectedIdInput, getHistoryRemoveSelectedButton);
        const isPreview = historyItemIsPreview(item);
        item.classList.toggle('history-selected-has-quality', isPreview);
        if (isPreview) {
            ensureHistorySelectedActionButton(item, 'history-selected-quality', 'Quality 60', 'Generate this preview at Quality, 60 steps', imageId,
                getHistoryQualitySelectedIdInput, getHistoryQualitySelectedButton);
        } else {
            item.querySelector('.history-selected-quality')?.remove();
        }
        ensureHistoryHideThumbnailButton(item, imageId, 'history-selected-hide');
    });
}

function installHistorySelectedMenuDismissal() {
    if (window.fooocusHistorySelectedMenuDismissalInstalled) return;
    window.fooocusHistorySelectedMenuDismissalInstalled = true;
    document.addEventListener('click', function(event) {
        if (event.target.closest?.('#history_selected_gallery .history-selected-image-item')) return;
        closeHistorySelectedMenus();
    });
}

function installHistoryDaySelectionMode() {
    const daySelection = gradioApp().querySelector('#history_day_selection');
    if (!daySelection) return;
    daySelection.querySelectorAll('label, input').forEach(function(item) {
        if (item.dataset.historyDaySelectionInstalled === 'true') return;
        item.dataset.historyDaySelectionInstalled = 'true';
        item.addEventListener('pointerdown', function(event) {
            setHistoryDaySelectionMode(event);
        }, true);
        item.addEventListener('click', function(event) {
            setHistoryDaySelectionMode(event);
        }, true);
    });
}

function installQueueButtons() {
    const removeButton = gradioButton('#remove_queued_task_button');
    const stopButton = gradioButton('#stop_queue_button');
    const skipButton = gradioButton('#skip_button');
    const panel = gradioApp().querySelector('#queue_status_panel');
    if (!removeButton || !stopButton || !skipButton || !panel) return;

    panel.querySelectorAll('.queue-remove-button').forEach(function(button) {
        if (button.dataset.boundQueueRemove) return;
        button.dataset.boundQueueRemove = 'true';
        button.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            if (setQueueRemoveId(button.dataset.queueId || '')) {
                removeButton.click();
            }
        };
    });

    panel.querySelectorAll('.queue-stop-button').forEach(function(button) {
        if (button.dataset.boundQueueStop) return;
        button.dataset.boundQueueStop = 'true';
        button.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            stopButton.click();
        };
    });

    panel.querySelectorAll('.queue-skip-button').forEach(function(button) {
        if (button.dataset.boundQueueSkip) return;
        button.dataset.boundQueueSkip = 'true';
        button.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            skipButton.click();
        };
    });
}

function isElementVisible(element) {
    if (!element) return false;
    const style = window.getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden' && element.offsetParent !== null;
}

function generationPollNeeded() {
    const panel = gradioApp().querySelector('#queue_status_panel');
    const hasQueueRows = !!panel?.querySelector('.queue-row');
    const progress = gradioApp().querySelector('#progress-bar');
    const reconnectButton = gradioButton('#reset_button');
    return hasQueueRows || isElementVisible(progress) || isElementVisible(reconnectButton);
}

function installGenerationPoller() {
    if (window.fooocusGenerationPollerInstalled) return;
    window.fooocusGenerationPollerInstalled = true;
    window.setInterval(function() {
        if (!generationPollNeeded()) return;
        gradioButton('#poll_generate_button')?.click();
    }, 1000);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;
var uiInstallerTimeout = null;

function runUiInstallers() {
    installPersonLikenessRemoveButtons();
    installGenerationHistoryApplyButtons();
    installHistoryThumbnailSelection();
    installHistorySelectedRemoveButtons();
    installHistorySelectedMenuDismissal();
    installHistoryDaySelectionMode();
    installQueueButtons();
}

function scheduleUiInstallers() {
    clearTimeout(uiInstallerTimeout);
    uiInstallerTimeout = setTimeout(runUiInstallers, 100);
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
        scheduleUiInstallers();
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    scheduleUiInstallers();
    installGenerationPoller();
    initStylePreviewOverlay();
});

var onAppend = function(elem, f) {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.addedNodes.length) {
                f(m.addedNodes);
            }
        });
    });
    observer.observe(elem, {childList: true});
}

function addObserverIfDesiredNodeAvailable(querySelector, callback) {
    var elem = document.querySelector(querySelector);
    if (!elem) {
        window.setTimeout(() => addObserverIfDesiredNodeAvailable(querySelector, callback), 1000);
        return;
    }

    onAppend(elem, callback);
}

/**
 * Show reset button on toast "Connection errored out."
 */
addObserverIfDesiredNodeAvailable(".toast-wrap", function(added) {
    added.forEach(function(element) {
         if (element.innerText.includes("Connection errored out.")) {
             window.setTimeout(function() {
                document.getElementById("reset_button")?.classList.remove("hidden");
                document.getElementById("skip_button")?.classList.add("hidden");
                document.getElementById("stop_button")?.classList.add("hidden");
            });
         }
    });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

function initStylePreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preview-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'stylePreviewOverlay';
    document.body.appendChild(overlay);
    document.addEventListener('mouseover', function (e) {
        const label = e.target.closest('.style_selections label');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        overlayVisible = true;
        overlay.style.opacity = "1";
        const originalText = label.querySelector("span").getAttribute("data-original-text");
        const name = originalText || label.querySelector("span").textContent;
        overlay.style.backgroundImage = `url("${samplesPath.replace(
            "fooocus_v2",
            name.toLowerCase().replaceAll(" ", "_")
        ).replaceAll("\\", "\\\\")}")`;

        tooltip.textContent = name;

        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function htmlDecode(input) {
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}
