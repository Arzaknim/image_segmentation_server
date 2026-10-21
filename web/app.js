document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileStrip = document.getElementById('fileStrip');
  const thumbImg = document.getElementById('thumbImg');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const runBtn = document.getElementById('runBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('spinner');

  const inputTileSize = document.getElementById('inputTileSize');
  const valTileSize = document.getElementById('valTileSize');
  const inputStride = document.getElementById('inputStride');
  const valStride = document.getElementById('valStride');

  const viewModeGroup = document.getElementById('viewModeGroup');
  const overlaySliderCtrl = document.getElementById('overlaySliderCtrl');
  const overlayOpacity = document.getElementById('overlayOpacity');
  const downloadBtn = document.getElementById('downloadBtn');

  const emptyState = document.getElementById('emptyState');
  const imageContainer = document.getElementById('imageContainer');
  const resultsGrid = document.getElementById('resultsGrid');
  const timingList = document.getElementById('timingList');
  const classLegendList = document.getElementById('classLegendList');

  let currentFile = null;
  let originalDataUrl = null;
  let maskDataUrl = null;
  let currentMode = 'side';

  // Slider bindings
  inputTileSize.addEventListener('input', (e) => {
    valTileSize.textContent = e.target.value;
    if (parseInt(inputStride.value, 10) > parseInt(e.target.value, 10)) {
      inputStride.value = e.target.value;
      valStride.textContent = e.target.value;
    }
  });

  inputStride.addEventListener('input', (e) => {
    if (parseInt(e.target.value, 10) > parseInt(inputTileSize.value, 10)) {
      e.target.value = inputTileSize.value;
    }
    valStride.textContent = e.target.value;
  });

  // Dropzone interactions
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  });

  function handleFileSelect(file) {
    currentFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
    fileStrip.style.display = 'flex';

    const reader = new FileReader();
    reader.onload = (e) => {
      originalDataUrl = e.target.result;
      thumbImg.src = originalDataUrl;
      maskDataUrl = null;
      renderView();
    };
    reader.readAsDataURL(file);

    runBtn.disabled = false;
    btnText.textContent = 'Run Segmentation';
  }

  // View mode switching
  viewModeGroup.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') {
      viewModeGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentMode = e.target.dataset.mode;
      overlaySliderCtrl.style.display = (currentMode === 'overlay') ? 'flex' : 'none';
      renderView();
    }
  });

  overlayOpacity.addEventListener('input', (e) => {
    const mask = document.getElementById('overlayMask');
    if (mask) mask.style.opacity = e.target.value;
  });

  function renderView() {
    if (!originalDataUrl && !maskDataUrl) {
      emptyState.style.display = 'block';
      imageContainer.style.display = 'none';
      return;
    }

    emptyState.style.display = 'none';
    imageContainer.style.display = 'flex';

    if (!maskDataUrl) {
      imageContainer.innerHTML = `
        <div class="view-box" style="max-width: 650px; width: 100%;">
          <div class="view-box-header">Original Aerial Image Preview (Ready for Inference)</div>
          <img src="${originalDataUrl}" alt="Original Preview">
        </div>
      `;
      return;
    }

    if (currentMode === 'side') {
      imageContainer.innerHTML = `
        <div class="view-side">
          <div class="view-box">
            <div class="view-box-header">Original Satellite Image</div>
            <img src="${originalDataUrl}" alt="Original">
          </div>
          <div class="view-box">
            <div class="view-box-header">Predicted Segmentation Mask</div>
            <img src="${maskDataUrl}" alt="Mask">
          </div>
        </div>
      `;
    } else if (currentMode === 'overlay') {
      imageContainer.innerHTML = `
        <div class="view-overlay-wrap">
          <img class="base-img" src="${originalDataUrl}" alt="Original">
          <img class="mask-img" id="overlayMask" src="${maskDataUrl}" style="opacity: ${overlayOpacity.value}" alt="Overlay Mask">
        </div>
      `;
    } else if (currentMode === 'mask') {
      imageContainer.innerHTML = `
        <div class="view-box" style="max-width: 700px; width: 100%;">
          <div class="view-box-header">Predicted Segmentation Mask</div>
          <img src="${maskDataUrl}" alt="Mask Only">
        </div>
      `;
    }
  }

  // Segmentation API call
  runBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    runBtn.disabled = true;
    spinner.style.display = 'block';
    btnText.textContent = 'Segmenting Tiles...';

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('tile_size', inputTileSize.value);
    formData.append('stride', inputStride.value);

    try {
      const response = await fetch('/api/segment', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Server error occurred during segmentation');
      }

      maskDataUrl = data.mask_png_base64;
      downloadBtn.href = maskDataUrl;
      downloadBtn.download = data.filename.replace(/\.[^/.]+$/, "") + "_mask.png";
      downloadBtn.style.display = 'inline-flex';

      renderView();
      populateResults(data);
    } catch (err) {
      alert('Segmentation Failed: ' + err.message);
    } finally {
      runBtn.disabled = false;
      spinner.style.display = 'none';
      btnText.textContent = 'Run Segmentation';
    }
  });

  function populateResults(data) {
    resultsGrid.style.display = 'grid';

    const t = data.timing;
    timingList.innerHTML = `
      <div class="metric-row"><span>Total Inference:</span><span>${(t.total_inference_time_ms / 1000).toFixed(3)}s (${t.total_inference_time_ms.toFixed(1)} ms)</span></div>
      <div class="metric-row"><span>Pure ONNX Forward:</span><span>${(t.pure_onnx_forward_ms / 1000).toFixed(3)}s</span></div>
      <div class="metric-row"><span>Computed Tiles:</span><span>${t.tile_count} tiles</span></div>
      <div class="metric-row"><span>Avg Tile Forward:</span><span>${t.avg_tile_forward_ms.toFixed(1)} ms/tile</span></div>
      <div class="metric-row"><span>Pre/Post Processing:</span><span>${t.pre_post_processing_ms.toFixed(1)} ms</span></div>
      <div class="metric-row"><span>Image Resolution:</span><span>${data.width} &times; ${data.height} px</span></div>
      ${data.log_file ? `<div class="metric-row"><span>JSON Log:</span><span style="font-size:0.75rem">${data.log_file}</span></div>` : ''}
      ${data.iou ? `
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(255,255,255,0.15)">
          <div class="metric-row" style="font-weight:bold; color:#4ade80"><span>Mean IoU (mIoU):</span><span>${data.iou.mean_iou.toFixed(4)}</span></div>
          ${Object.entries(data.iou.per_class).map(([cls, val]) => `
            <div class="metric-row" style="font-size:0.8rem; padding-left: 8px">
              <span>${cls}:</span>
              <span>${val !== null && val !== undefined ? Number(val).toFixed(4) : 'N/A (absent)'}</span>
            </div>
          `).join('')}
        </div>
      ` : ''}
    `;

    classLegendList.innerHTML = data.classes.map(c => `
      <div class="legend-item">
        <div class="legend-header">
          <div class="legend-tag">
            <div class="color-dot" style="background: ${c.hex_color}"></div>
            <span>${c.name}</span>
          </div>
          <span><b>${c.percentage.toFixed(2)}%</b> (${c.pixel_count.toLocaleString()} px)</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${c.percentage}%; background: ${c.hex_color}"></div>
        </div>
      </div>
    `).join('');
  }
});

