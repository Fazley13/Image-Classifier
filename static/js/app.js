(() => {
  "use strict";

  const MAX_BYTES = 16 * 1024 * 1024; // keep in sync with MAX_UPLOAD_MB in app.py

  const $ = (selector) => document.querySelector(selector);
  const els = {
    input: $("#fileInput"),
    slide: $("#slide"),
    dropBtn: $("#dropBtn"),
    photo: $("#photo"),
    preview: $("#preview"),
    toolbar: $("#toolbar"),
    info: $("#photoInfo"),
    cropToggle: $("#cropToggle"),
    addBtn: $("#addBtn"),
    clearBtn: $("#clearBtn"),
    strip: $("#strip"),
    notice: $("#notice"),
    results: $("#results"),
    status: $("#modelStatus"),
    statusText: $("#modelStatusText"),
  };

  const state = {
    items: [],        // { id, file, url, width, height, status, predictions, error }
    activeId: null,
    running: false,
    modelState: "loading",
  };

  let nextId = 1;
  let noticeTimer = null;

  // ---- Small DOM helper ----------------------------------------------------
  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === false || value == null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(child));
    }
    return node;
  }

  const activeItem = () => state.items.find((item) => item.id === state.activeId) || null;
  const percent = (value) => `${(value * 100).toFixed(1)}%`;

  // ---- Notices -------------------------------------------------------------
  function notify(message) {
    els.notice.textContent = message;
    els.notice.hidden = false;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { els.notice.hidden = true; }, 7000);
  }

  // ---- Model status --------------------------------------------------------
  const STATUS_TEXT = {
    idle: "Starting model",
    loading: "Loading model",
    ready: "Model ready",
    error: "Model unavailable",
    offline: "Server not reachable",
  };

  function setModelState(next, detail) {
    state.modelState = next;
    els.status.dataset.state = next === "idle" ? "loading" : next;
    els.statusText.textContent = STATUS_TEXT[next] || next;
    els.status.title = detail || "";
    if (activeItem()) renderResults();
  }

  async function pollHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      const data = await response.json();
      setModelState(data.state, data.error);
      if (data.state === "idle" || data.state === "loading") setTimeout(pollHealth, 1000);
    } catch {
      setModelState("offline");
      setTimeout(pollHealth, 3000);
    }
  }

  // ---- Adding files --------------------------------------------------------
  function readDimensions(url) {
    return new Promise((resolve, reject) => {
      const probe = new Image();
      probe.onload = () => resolve({ width: probe.naturalWidth, height: probe.naturalHeight });
      probe.onerror = () => reject(new Error("unreadable"));
      probe.src = url;
    });
  }

  async function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    const skipped = [];
    let firstAdded = null;

    for (const file of files) {
      if (!file.type.startsWith("image/")) {
        skipped.push(`${file.name} is not an image`);
        continue;
      }
      if (file.size > MAX_BYTES) {
        skipped.push(`${file.name} is larger than 16 MB`);
        continue;
      }
      const url = URL.createObjectURL(file);
      try {
        const { width, height } = await readDimensions(url);
        const item = {
          id: nextId++, file, url, width, height,
          status: "queued", predictions: null, error: null,
        };
        state.items.push(item);
        firstAdded = firstAdded || item;
      } catch {
        URL.revokeObjectURL(url);
        skipped.push(`${file.name} can't be opened in the browser`);
      }
    }

    if (skipped.length) notify(`Skipped: ${skipped.join("; ")}.`);
    if (firstAdded) state.activeId = firstAdded.id;
    renderAll();
    processQueue();
  }

  // ---- Prediction queue ----------------------------------------------------
  async function requestPrediction(item) {
    const body = new FormData();
    body.append("image", item.file, item.file.name);

    let response;
    try {
      response = await fetch("/api/predict", { method: "POST", body });
    } catch {
      throw new Error("Couldn't reach the local server. Check that app.py is still running.");
    }

    let data = null;
    try { data = await response.json(); } catch { /* non-JSON error page */ }

    if (!response.ok) {
      const failure = new Error((data && data.error) || `The server returned an error (${response.status}).`);
      failure.modelProblem = response.status === 503;
      throw failure;
    }
    return data.predictions;
  }

  async function processQueue() {
    if (state.running) return;
    state.running = true;
    try {
      let item;
      while ((item = state.items.find((entry) => entry.status === "queued"))) {
        item.status = "loading";
        renderAll();
        try {
          item.predictions = await requestPrediction(item);
          item.status = "done";
        } catch (error) {
          item.status = "error";
          item.error = error.message;
          item.modelProblem = Boolean(error.modelProblem);
          pollHealth();
        }
        renderAll();
      }
    } finally {
      state.running = false;
    }
  }

  // ---- Rendering: stage ----------------------------------------------------
  function fitPhoto() {
    const item = activeItem();
    if (!item) return;
    const style = getComputedStyle(els.slide);
    const maxW = els.slide.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const maxH = els.slide.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    const scale = Math.min(maxW / item.width, maxH / item.height);
    const width = Math.max(1, Math.floor(item.width * scale));
    const height = Math.max(1, Math.floor(item.height * scale));
    els.photo.style.width = `${width}px`;
    els.photo.style.height = `${height}px`;
    els.photo.style.setProperty("--crop", `${Math.min(width, height)}px`);
  }

  function renderStage() {
    const item = activeItem();
    const hasItems = state.items.length > 0;

    els.slide.dataset.empty = String(!item);
    els.photo.hidden = !item;
    els.toolbar.hidden = !hasItems;
    els.strip.hidden = state.items.length < 2;

    if (!item) return;

    if (els.preview.src !== item.url) {
      els.preview.src = item.url;
      els.preview.alt = `Preview of ${item.file.name}`;
    }
    els.photo.dataset.busy = String(item.status === "loading");
    els.info.replaceChildren(
      h("strong", { text: item.file.name }),
      ` — ${item.width} × ${item.height} px`,
    );
    fitPhoto();
  }

  function renderStrip() {
    els.strip.replaceChildren(
      ...state.items.map((item) => {
        const badge = { done: "✓", error: "!", loading: "…", queued: "" }[item.status];
        return h("li", {},
          h("button", {
            type: "button",
            class: "thumb",
            "data-status": item.status,
            "aria-current": String(item.id === state.activeId),
            "aria-label": `${item.file.name}, ${item.status === "done" ? "analyzed" : item.status}`,
            title: item.file.name,
            onclick: () => { state.activeId = item.id; renderAll(); },
          },
            h("img", { src: item.url, alt: "" }),
            badge ? h("span", { class: "badge", "aria-hidden": "true", text: badge }) : null,
          ),
        );
      }),
    );
  }

  // ---- Rendering: results --------------------------------------------------
  function bar(value, extraClass = "") {
    const fill = h("i");
    const track = h("div", { class: `bar ${extraClass}`.trim(), role: "presentation" }, fill);
    // set the width on the next frame so the bar animates in
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = `${Math.max(value * 100, 1)}%`;
    }));
    return track;
  }

  function verdictFor(confidence) {
    if (confidence >= 0.7) return ["high", "A strong match."];
    if (confidence >= 0.35) return ["mid", "Likely, but the model isn't certain. Check the other possibilities below."];
    return ["low", "Low confidence. The subject may be small, partly hidden, or outside the 1,000 ImageNet classes."];
  }

  function renderIntro() {
    return [
      h("h1", { class: "intro-title", text: "What's in this image?" }),
      h("p", {
        class: "intro-text",
        text: "Add a photo and MobileNetV2 names the most likely objects, with a confidence score for each guess.",
      }),
      h("ul", { class: "tips" },
        h("li", { text: "It knows 1,000 everyday classes: animals, vehicles, tools, food, and more." }),
        h("li", { text: "It works best when one object is clear and near the center." }),
        h("li", { text: "Add several images at once and switch between them below the slide." }),
      ),
    ];
  }

  function renderWorking(item) {
    const loadingModel = state.modelState === "loading" || state.modelState === "idle";
    return [
      h("p", { class: "file-name", text: item.file.name }),
      h("h2", { class: "work-title", text: loadingModel ? "Loading the model" : "Analyzing" }),
      h("p", {
        class: "work-text",
        text: loadingModel
          ? "The first run downloads about 14 MB of ImageNet weights, so this can take a minute."
          : "Looking at the center of the image.",
      }),
      h("div", { class: "skeleton", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
    ];
  }

  function renderError(item) {
    return [
      h("p", { class: "file-name", text: item.file.name }),
      h("h2", {
        class: "error-title",
        text: item.modelProblem ? "The model isn't available" : "This image couldn't be analyzed",
      }),
      h("p", { class: "error-text", text: item.error }),
      h("button", {
        type: "button",
        class: "btn",
        onclick: () => { item.status = "queued"; item.error = null; renderAll(); processQueue(); },
        text: "Try again",
      }),
    ];
  }

  function renderDone(item) {
    const [best, ...others] = item.predictions;
    const [level, message] = verdictFor(best.confidence);
    return [
      h("p", { class: "file-name", text: item.file.name }),
      h("h2", { class: "top-label" }, h("span", { text: best.label })),
      h("p", { class: "top-confidence" },
        h("strong", { text: percent(best.confidence) }),
        h("span", { text: "confidence" }),
      ),
      bar(best.confidence, "big"),
      h("p", { class: "verdict", "data-level": level, text: message }),
      others.length
        ? [
            h("h3", { class: "alts-title", text: "Other possibilities" }),
            h("ol", { class: "alts", start: "2" },
              others.map((p) =>
                h("li", {},
                  h("span", { class: "rank", text: `${p.rank}.` }),
                  h("span", { class: "name", text: p.label }),
                  h("span", { class: "pct", text: percent(p.confidence) }),
                  bar(p.confidence),
                ),
              ),
            ),
          ]
        : null,
    ];
  }

  let resultsKey = null; // skip re-rendering when nothing visible changed (keeps bar animations calm)

  function renderResults() {
    const item = activeItem();
    const working = item && (item.status === "loading" || item.status === "queued");
    const key = item ? `${item.id}:${item.status}:${working ? state.modelState : ""}` : "intro";
    if (key === resultsKey) return;
    resultsKey = key;

    let content;
    if (!item) content = renderIntro();
    else if (item.status === "done") content = renderDone(item);
    else if (item.status === "error") content = renderError(item);
    else content = renderWorking(item);

    els.results.setAttribute("aria-busy", String(!!item && (item.status === "loading" || item.status === "queued")));
    els.results.replaceChildren(...content.flat().filter(Boolean));
  }

  function renderAll() {
    renderStage();
    renderStrip();
    renderResults();
  }

  // ---- Clearing ------------------------------------------------------------
  function clearAll() {
    state.items.forEach((item) => URL.revokeObjectURL(item.url));
    state.items = [];
    state.activeId = null;
    els.preview.removeAttribute("src");
    els.input.value = "";
    renderAll();
  }

  // ---- Events --------------------------------------------------------------
  els.dropBtn.addEventListener("click", () => els.input.click());
  els.addBtn.addEventListener("click", () => els.input.click());
  els.clearBtn.addEventListener("click", clearAll);
  els.input.addEventListener("change", () => {
    addFiles(els.input.files);
    els.input.value = ""; // allows choosing the same file again
  });

  els.cropToggle.addEventListener("click", () => {
    const on = els.cropToggle.getAttribute("aria-pressed") !== "true";
    els.cropToggle.setAttribute("aria-pressed", String(on));
    els.slide.dataset.crop = on ? "on" : "off";
  });

  // Drag and drop anywhere on the page.
  let dragDepth = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");

  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth++;
    els.slide.dataset.drag = "true";
  });
  window.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault(); // required so the drop is allowed
  });
  window.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) els.slide.dataset.drag = "false";
  });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault(); // stops the browser from navigating to the file
    dragDepth = 0;
    els.slide.dataset.drag = "false";
    addFiles(event.dataTransfer.files);
  });

  // Paste images from the clipboard.
  window.addEventListener("paste", (event) => {
    const files = Array.from(event.clipboardData?.files || []).filter((f) => f.type.startsWith("image/"));
    if (files.length) {
      event.preventDefault();
      addFiles(files);
    }
  });

  new ResizeObserver(fitPhoto).observe(els.slide);

  // ---- Start ---------------------------------------------------------------
  renderAll();
  pollHealth();
})();
