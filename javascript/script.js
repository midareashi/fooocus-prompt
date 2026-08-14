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

function getGenerationFavoriteButton() {
    return gradioApp().querySelector('#favorite_selected_generation_button');
}

function getQuickPreviewGenerationIndicesInput() {
    return gradioApp().querySelector(
        '#quick_preview_generation_indices textarea, #quick_preview_generation_indices input, textarea#quick_preview_generation_indices, input#quick_preview_generation_indices'
    );
}

function setGenerationActionIndex(input, index) {
    if (!input) return false;
    const value = String(index);
    const valueSetter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), 'value')?.set;
    if (valueSetter) {
        valueSetter.call(input, value);
    } else {
        input.value = value;
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
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

function getQueueRemoveIdInput() {
    return gradioApp().querySelector(
        '#selected_queue_remove_id textarea, #selected_queue_remove_id input, textarea#selected_queue_remove_id, input#selected_queue_remove_id'
    );
}

function setQueueRemoveId(queueId) {
    return setGenerationActionIndex(getQueueRemoveIdInput(), queueId);
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
    button.setAttribute('aria-label', title);
    button.setAttribute('title', title);
    button.onclick = onClick;
}

function setFavoriteStarState(button, isFavorite) {
    if (!button) return;
    button.classList.toggle('history-favorite-active', isFavorite);
    button.textContent = isFavorite ? '\u2605' : '\u2606';
    button.setAttribute('aria-label', isFavorite ? 'Remove favorite' : 'Add favorite');
    button.setAttribute('title', isFavorite ? 'Remove favorite' : 'Add favorite');
}

function installGenerationHistoryApplyButtons() {
    const gallery = gradioApp().querySelector('#final_gallery');
    const applyButton = gradioApp().querySelector('#apply_selected_image_config_button');
    const removeButton = gradioApp().querySelector('#remove_selected_image_button');
    const deleteButton = gradioApp().querySelector('#delete_selected_image_button');
    const qualityButton = gradioApp().querySelector('#regenerate_selected_quality_button');
    const favoriteButton = getGenerationFavoriteButton();
    const previewIndicesInput = getQuickPreviewGenerationIndicesInput();
    if (!gallery || !applyButton || !removeButton || !deleteButton || !qualityButton || !favoriteButton) return;

    let previewIndices = [];
    try {
        previewIndices = JSON.parse(previewIndicesInput?.value || '[]');
    } catch {
        previewIndices = [];
    }
    const previewIndexSet = new Set(previewIndices.map((index) => Number(index)));

    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item, index) {
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
        ensureGenerationHistoryButton(item, 'generation-history-favorite', '★', 'Favorite image', function(event) {
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

function getHistoryDaySelectionModeInput() {
    return gradioApp().querySelector('#history_day_selection_mode textarea, #history_day_selection_mode input');
}

function getHistorySelectedIdsInput() {
    return gradioApp().querySelector('#history_selected_image_ids_json textarea, #history_selected_image_ids_json input');
}

function getHistoryRemoveSelectedIdInput() {
    return gradioApp().querySelector('#history_remove_selected_image_id textarea, #history_remove_selected_image_id input');
}

function getHistoryRemoveSelectedButton() {
    return gradioApp().querySelector('#history_remove_selected_image_button');
}

function getHistoryDeleteSelectedIdInput() {
    return gradioApp().querySelector('#history_delete_selected_image_id textarea, #history_delete_selected_image_id input');
}

function getHistoryDeleteSelectedButton() {
    return gradioApp().querySelector('#history_delete_selected_image_button');
}

function getHistoryApplySelectedIdInput() {
    return gradioApp().querySelector('#history_apply_selected_image_id textarea, #history_apply_selected_image_id input');
}

function getHistoryApplySelectedButton() {
    return gradioApp().querySelector('#history_apply_selected_image_button');
}

function getHistoryToggleFavoriteIdInput() {
    return gradioApp().querySelector('#history_toggle_favorite_image_id textarea, #history_toggle_favorite_image_id input');
}

function getHistoryToggleFavoriteButton() {
    return gradioApp().querySelector('#history_toggle_favorite_button');
}

function setHistorySelectionMode(event) {
    const input = getHistorySelectionModeInput();
    if (!input) return;
    let mode = 'single';
    if (event.shiftKey) {
        mode = 'shift';
    } else if (event.ctrlKey || event.metaKey) {
        mode = 'ctrl';
    }
    input.value = mode;
    input.dispatchEvent(new Event('input', { bubbles: true }));
}

function setHistoryDaySelectionMode(event) {
    const input = getHistoryDaySelectionModeInput();
    if (!input) return;
    let mode = 'single';
    if (event.shiftKey) {
        mode = 'shift';
    } else if (event.ctrlKey || event.metaKey) {
        mode = 'ctrl';
    }
    input.value = mode;
    input.dispatchEvent(new Event('input', { bubbles: true }));
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

function historyItemCaptionText(item) {
    return item.querySelector('.caption, .thumbnail-label, p, span')?.textContent || item.textContent || '';
}

function historyItemIsFavorite(item) {
    return /\bfav\b/i.test(historyItemCaptionText(item));
}

function triggerHistoryFavoriteToggle(imageId) {
    const input = getHistoryToggleFavoriteIdInput();
    const button = getHistoryToggleFavoriteButton();
    if (!input || !button || imageId === null) return false;
    input.value = String(imageId);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    button.click();
    return true;
}

function triggerHistoryImageAction(input, button, imageId) {
    if (!input || !button || imageId === null) return false;
    input.value = String(imageId);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    button.click();
    return true;
}

function ensureHistoryFavoriteButton(item, imageId) {
    let button = item.querySelector('.history-favorite-toggle');
    if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'history-favorite-toggle';
        button.textContent = '★';
        item.appendChild(button);
    }
    const isFavorite = historyItemIsFavorite(item);
    button.classList.toggle('history-favorite-active', isFavorite);
    button.setAttribute('aria-label', isFavorite ? 'Remove favorite' : 'Add favorite');
    button.setAttribute('title', isFavorite ? 'Remove favorite' : 'Add favorite');
    setFavoriteStarState(button, isFavorite);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        if (triggerHistoryFavoriteToggle(imageId)) {
            const nextFavorite = !button.classList.contains('history-favorite-active');
            button.classList.toggle('history-favorite-active', nextFavorite);
            button.setAttribute('aria-label', nextFavorite ? 'Remove favorite' : 'Add favorite');
            button.setAttribute('title', nextFavorite ? 'Remove favorite' : 'Add favorite');
            setFavoriteStarState(button, nextFavorite);
        }
    };
}

function installHistoryThumbnailSelection() {
    const gallery = gradioApp().querySelector('#history_thumbnail_gallery');
    if (!gallery) return;
    const selectedIds = getHistorySelectedIds();
    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item) {
        if (item.dataset.historySelectionInstalled !== 'true') {
            item.dataset.historySelectionInstalled = 'true';
            item.addEventListener('pointerdown', function(event) {
                setHistorySelectionMode(event);
            }, true);
            item.addEventListener('click', function(event) {
                setHistorySelectionMode(event);
            }, true);
        }
        const imageId = getHistoryImageIdFromThumb(item);
        ensureHistoryFavoriteButton(item, imageId);
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
    button.setAttribute('aria-label', title);
    button.setAttribute('title', title);
    button.onclick = function(event) {
        event.preventDefault();
        event.stopPropagation();
        triggerHistoryImageAction(inputGetter(), buttonGetter(), imageId);
    };
}

function installHistorySelectedRemoveButtons() {
    const gallery = gradioApp().querySelector('#history_selected_gallery');
    if (!gallery) return;
    const items = Array.from(gallery.querySelectorAll('.thumbnail-item'));
    items.forEach(function(item) {
        item.classList.add('history-selected-image-item');
        const imageId = getHistoryImageIdFromThumb(item);
        ensureHistoryFavoriteButton(item, imageId);
        ensureHistorySelectedActionButton(item, 'history-selected-apply', 'Apply Config', 'Apply image config', imageId,
            getHistoryApplySelectedIdInput, getHistoryApplySelectedButton);
        ensureHistorySelectedActionButton(item, 'history-selected-remove', 'Remove', 'Remove from selected images', imageId,
            getHistoryRemoveSelectedIdInput, getHistoryRemoveSelectedButton);
        ensureHistorySelectedActionButton(item, 'history-selected-delete', 'Delete', 'Delete image file and history record', imageId,
            getHistoryDeleteSelectedIdInput, getHistoryDeleteSelectedButton);
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
    const removeButton = gradioApp().querySelector('#remove_queued_task_button');
    const stopButton = gradioApp().querySelector('#stop_queue_button');
    const panel = gradioApp().querySelector('#queue_status_panel');
    if (!removeButton || !stopButton || !panel) return;

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
        installPersonLikenessRemoveButtons();
        installGenerationHistoryApplyButtons();
        installHistoryThumbnailSelection();
        installHistorySelectedRemoveButtons();
        installHistoryDaySelectionMode();
        installQueueButtons();
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
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
