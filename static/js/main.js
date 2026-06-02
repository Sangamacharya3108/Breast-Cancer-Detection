(function () {
  "use strict";

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  /* ---------- Predict page: file input, drag-drop, preview ---------- */
  var dropZone = $("#dropZone");
  var imageInput = $("#imageInput");
  var fileNameEl = $("#fileName");
  var imagePreview = $("#imagePreview");
  var previewPlaceholder = $("#previewPlaceholder");
  var submitBtn = $("#submitBtn");
  var predictForm = $("#predictForm");
  var globalLoading = $("#globalLoading");

  function setFile(file) {
    if (!file || !imageInput) return;
    var dt = new DataTransfer();
    dt.items.add(file);
    imageInput.files = dt.files;
    if (fileNameEl) fileNameEl.textContent = file.name;
    if (submitBtn) submitBtn.disabled = false;

    if (imagePreview && previewPlaceholder) {
      var reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove("d-none");
        previewPlaceholder.classList.add("d-none");
      };
      reader.readAsDataURL(file);
    }
  }

  if (dropZone && imageInput) {
    dropZone.addEventListener("click", function () {
      imageInput.click();
    });

    imageInput.addEventListener("change", function () {
      var f = imageInput.files && imageInput.files[0];
      if (f) setFile(f);
    });

    ["dragenter", "dragover", "dragleave", "drop"].forEach(function (ev) {
      dropZone.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ["dragenter", "dragover"].forEach(function (ev) {
      dropZone.addEventListener(ev, function () {
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(function (ev) {
      dropZone.addEventListener(ev, function () {
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", function (e) {
      var files = e.dataTransfer.files;
      if (files && files[0] && files[0].type.indexOf("image/") === 0) {
        setFile(files[0]);
      }
    });

    dropZone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        imageInput.click();
      }
    });
  }

  /* ---------- Form submit: loading overlay ---------- */
  if (predictForm && globalLoading) {
    predictForm.addEventListener("submit", function () {
      if (predictForm.getAttribute("data-loading") === "true") {
        globalLoading.classList.remove("d-none");
      }
    });
  }

  /* If page loads with result, ensure overlay hidden */
  if (globalLoading) {
    globalLoading.classList.add("d-none");
  }

  /* ---------- Animate confidence progress bar on result load ---------- */
  var progressBar = document.querySelector(".progress-bar[data-target-width]");
  if (progressBar) {
    // Start from 0 and animate to the target width after a brief paint delay
    var targetWidth = progressBar.getAttribute("data-target-width");
    progressBar.style.width = "0%";
    setTimeout(function () {
      progressBar.style.width = targetWidth;
    }, 80);
  }
})();
