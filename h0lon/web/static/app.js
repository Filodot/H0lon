/* H0lon local web interface: small progressive enhancements, no libraries.
   Every page works without this file; it adds drag-and-drop upload, the live job log
   (Server-Sent Events), refresh of the status blocks when a job ends, and the review panes. */
(function () {
  "use strict";

  document.documentElement.classList.add("js");

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));

  // ---------------------------------------------------------------- toast

  let toastTimer = null;
  function toast(text, ms) {
    const el = $("[data-toast]");
    if (!el) return;
    el.textContent = text;
    el.hidden = false;
    clearTimeout(toastTimer);
    if (ms !== 0) toastTimer = setTimeout(() => { el.hidden = true; }, ms || 3500);
  }

  // ---------------------------------------------------------------- forms: confirm and busy state

  const busyElements = [];

  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    if (form.dataset.busy) {
      toast(form.dataset.busy, 0);
      // After the browser has built the form data: a disabled button would not be sent.
      setTimeout(() => {
        $$("button[type=submit], button:not([type])", form).forEach((button) => {
          if (button.disabled) return;
          button.disabled = true;
          button.classList.add("is-busy");
          busyElements.push(button);
        });
      }, 0);
    }
  });

  document.addEventListener("click", (event) => {
    const link = event.target.closest && event.target.closest("a[data-busy-link]");
    if (link && !event.defaultPrevented && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
      toast(link.dataset.busyLink, 0);
    }
  });

  // Back/forward cache: forget the busy state of the page we come back to.
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    busyElements.splice(0).forEach((button) => {
      button.disabled = false;
      button.classList.remove("is-busy");
    });
    const el = $("[data-toast]");
    if (el) el.hidden = true;
  });

  // ---------------------------------------------------------------- drag and drop upload

  function humanSize(bytes) {
    if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " КБ";
    return (bytes / (1024 * 1024)).toFixed(1).replace(".", ",") + " МБ";
  }

  function initUpload() {
    const form = $("[data-add-form]");
    if (!form) return;
    const input = $("input[type=file]", form);
    const zone = $("[data-dropzone]", form);
    const list = $("[data-file-list]", form);
    const overlay = $("[data-drop-overlay]");
    if (!input || !zone || input.disabled) return;

    function showFiles() {
      list.replaceChildren(
        ...Array.from(input.files).map((file) => {
          const item = document.createElement("li");
          item.textContent = file.name + " (" + humanSize(file.size) + ")";
          return item;
        })
      );
    }
    input.addEventListener("change", showFiles);

    const hasFiles = (event) =>
      event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files");
    let depth = 0;

    window.addEventListener("dragenter", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth += 1;
      if (overlay) overlay.hidden = false;
      zone.classList.add("is-over");
    });
    window.addEventListener("dragover", (event) => {
      if (hasFiles(event)) event.preventDefault();
    });
    window.addEventListener("dragleave", (event) => {
      if (!hasFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) {
        if (overlay) overlay.hidden = true;
        zone.classList.remove("is-over");
      }
    });
    window.addEventListener("drop", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth = 0;
      if (overlay) overlay.hidden = true;
      zone.classList.remove("is-over");
      const items = Array.from(event.dataTransfer.items || []);
      const hasFolder = items.some((item) => {
        const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
        return entry && entry.isDirectory;
      });
      if (hasFolder) {
        toast("Папки не поддерживаются: перетащите сами файлы.", 5000);
        return;
      }
      if (!event.dataTransfer.files.length) return;
      input.files = event.dataTransfer.files;
      showFiles();
      form.requestSubmit();
    });
  }

  // ---------------------------------------------------------------- live job log

  function setBadge(panel, status, label) {
    const badge = $("[data-role=badge]", panel);
    if (!badge) return;
    badge.className = "badge badge-" + status;
    badge.textContent = label;
  }

  // Replace the status blocks of the topic page with their fresh versions from the server.
  async function refreshRegions() {
    const root = $("#topic");
    if (!root || !root.dataset.topicUrl) return location.reload();
    try {
      const response = await fetch(root.dataset.topicUrl, {
        credentials: "same-origin",
        headers: { Accept: "text/html" },
      });
      if (!response.ok) throw new Error(String(response.status));
      const fresh = new DOMParser().parseFromString(await response.text(), "text/html");
      $$("[data-refresh]").forEach((old) => {
        const next = $('[data-refresh="' + old.dataset.refresh + '"]', fresh);
        if (next) old.replaceWith(document.importNode(next, true));
      });
    } catch (error) {
      location.reload();
    }
  }

  function initJob() {
    const panel = $("#job");
    if (!panel) return;
    const log = $("[data-role=log]", panel);
    if (!log) return;

    let stick = true;
    log.addEventListener("scroll", () => {
      stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
    });
    log.scrollTop = log.scrollHeight;
    if (panel.dataset.jobActive !== "1" || !window.EventSource) return;

    let next = Number(panel.dataset.next) || 0;
    let hasText = log.textContent.length > 0;
    let lineCount = log.textContent ? log.textContent.split("\n").length : 0;
    let pending = [];
    let scheduled = false;

    function flush() {
      scheduled = false;
      if (!pending.length) return;
      const text = (hasText ? "\n" : "") + pending.join("\n");
      lineCount += pending.length;
      pending = [];
      hasText = true;
      log.appendChild(document.createTextNode(text));
      if (lineCount > 4000) {
        log.textContent = log.textContent.split("\n").slice(-3000).join("\n");
        lineCount = 3000;
      }
      if (stick) log.scrollTop = log.scrollHeight;
    }

    const source = new EventSource("/jobs/" + panel.dataset.jobId + "/events?from=" + next);
    source.onmessage = (event) => {
      pending.push(event.data);
      if (event.lastEventId) next = Number(event.lastEventId);
      if (!scheduled) {
        scheduled = true;
        requestAnimationFrame(flush);
      }
    };
    source.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      setBadge(panel, data.status, data.label);
    });
    source.addEventListener("done", async (event) => {
      source.close();
      flush();
      const data = JSON.parse(event.data);
      setBadge(panel, data.status, data.label);
      panel.dataset.jobActive = "0";
      await refreshRegions();
      const fresh = $("#job [data-role=log]");
      if (fresh) fresh.scrollTop = fresh.scrollHeight;
    });
    source.onerror = () => {
      // EventSource reconnects by itself (with Last-Event-ID); tell the user only if it gave up.
      if (source.readyState === EventSource.CLOSED) {
        toast("Связь с сервером потеряна. Обновите страницу.", 6000);
      }
    };
  }

  // ---------------------------------------------------------------- source review

  function initReview() {
    const root = $("[data-review]");
    if (!root) return;
    const tabs = $$("[data-pane-tab]", root);
    const narrow = () => window.matchMedia("(max-width: 900px)").matches;

    function showPane(name) {
      root.dataset.pane = name;
      tabs.forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.paneTab === name)));
    }
    tabs.forEach((tab) => tab.addEventListener("click", () => showPane(tab.dataset.paneTab)));

    const doc = $(".srcdoc", root);
    const toggle = $("[data-toggle-blocks]");
    if (toggle && doc) {
      toggle.addEventListener("change", () => doc.classList.toggle("show-blk", toggle.checked));
    }

    function scrollTo(target) {
      const body = target.closest(".pane-body");
      if (body && body.scrollHeight > body.clientHeight + 4) {
        const top = target.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;
        body.scrollTo({ top: Math.max(0, top - 8), behavior: "smooth" });
      } else {
        target.scrollIntoView({ block: "start", behavior: "smooth" });
      }
      target.classList.add("is-target");
      setTimeout(() => target.classList.remove("is-target"), 1600);
    }

    // A location heading of the Source Doc <-> the picture of that page.
    if (doc) {
      doc.addEventListener("click", (event) => {
        const heading = event.target.closest(".loc[data-page]");
        if (!heading || event.target.closest("a")) return;
        const figure = document.getElementById("pg-" + heading.dataset.page);
        if (!figure) {
          toast("Картинки этой страницы нет: она не отправлялась агенту. Смотрите исходный файл.");
          return;
        }
        if (narrow()) showPane("orig");
        scrollTo(figure);
      });
    }
    $$(".pageimg", root).forEach((figure) => {
      figure.addEventListener("click", () => {
        const heading = $('.loc[data-page="' + figure.dataset.page + '"]', root);
        if (!heading) return;
        if (narrow()) showPane("doc");
        scrollTo(heading);
      });
    });
  }

  initUpload();
  initJob();
  initReview();
})();
