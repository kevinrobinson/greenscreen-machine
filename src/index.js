// console.log('dependencies', {JSZip, _, tmImage, UMAP, ScatterGL});

const RANDOM_SEED = 42;
console.log('RANDOM_SEED', RANDOM_SEED);

function debug(...params) {
  //console.log(...params);
}

async function readBlobFromZip(zip, filename) {
  let zipEntry = null;
  zip.forEach((relativePath, entry) => {
    if (entry.name === filename) {
      zipEntry = entry;
    }
  });
  if (zipEntry === null) return;

  const fileData = await zipEntry.async('blob');
  return new File([fileData], filename);
}


async function loadImageModelFromZipFile(modelZipFile) {
  debug('Opening model zip...');
  const jsZip = new JSZip();
  const zip = await jsZip.loadAsync(modelZipFile);
  
  console.log('Loading model...');
  const model = await tmImage.loadFromFiles(
    await readBlobFromZip(zip, 'model.json'),
    await readBlobFromZip(zip, 'weights.bin'),
    await readBlobFromZip(zip, 'metadata.json')
  );
  console.log('Done.');
  return model;
}

// copied
function newCanvas() {
  return document.createElement('canvas');
}
function cropTo(image, size, flipped = false, canvas = null) {
    if (!canvas) canvas = newCanvas();

    // image image, bitmap, or canvas
    let width = image.width;
    let height = image.height;

    // if video element
    if (image instanceof HTMLVideoElement) {
        width = image.videoWidth;
        height = image.videoHeight;
    }

    const min = Math.min(width, height);
    const scale = size / min;
    const scaledW = Math.ceil(width * scale);
    const scaledH = Math.ceil(height * scale);
    const dx = scaledW - size;
    const dy = scaledH - size;
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, ~~(dx / 2) * -1, ~~(dy / 2) * -1, scaledW, scaledH);

    if (flipped) {
        ctx.scale(-1, 1);
        ctx.drawImage(canvas, scaledW * -1, 0);
    }

    return canvas;
}


async function embeddings(generalization, model, el, options = {}) {
  // clear, show waiting
  el.innerHTML = '';
  const waitingEl = document.createElement('div');
  waitingEl.classList.add('Status');
  waitingEl.textContent = 'working...';
  el.appendChild(waitingEl);

  debug('Starting embeddings...');
  const examples = await mapExamples(generalization, async (className, blobUrl, index) => {
    const imgEl = document.createElement('img');
    await setImageSrc(imgEl, blobUrl);
    const predictions = await model.predict(imgEl);
    return {className, index, predictions, blobUrl};
  });
  const embeddingsList = await mapExamples(generalization, async (className, blobUrl, index) => {
    const imgEl = document.createElement('img');
    await setImageSrc(imgEl, blobUrl);
    return (await infer(model, imgEl)).dataSync();
  });
  const baseEmbeddingsList = await mapExamples(generalization, async (className, blobUrl, index) => {
    const imgEl = document.createElement('img');
    await setImageSrc(imgEl, blobUrl);
    return (await inferMobileNet(model, imgEl)).dataSync();
  });

    //localStorage.setItem('examples', JSON.stringify(examples));localStorage.setItem('embeddingsList', JSON.stringify(embeddingsList));
  window.useProjector = useProjector;
  window.baseEmbeddingsList = baseEmbeddingsList;
  window.embeddingsList = embeddingsList;
  window.examples = examples;


  // grid in mobilenet space, mapped to TM space, to see warping
  const showGrid = false
  if (showGrid) {
    console.log('  grid points...');
    const vals = _.range(-5, 5, 1).map(n => n/100);
    const gridPoints = vals.map(v => _.range(0, 1280).map(i => v));
    const gridTransforms = await Promise.all(gridPoints.map(async p => {
      return (await inferFromMobileNetEmbedding(model, tf.tensor([p]))).dataSync();
    }));
    window.gridPoints = gridPoints;
    window.gridTransforms = gridTransforms;

    // useProjector(baseEl, baseEmbeddingsList, examples, gridPoints, options);
    // useProjector(trainedEl, embeddingsList, examples, gridTransforms, options);
  }


  debug('Projecting with UMAP...');
  // older: projectWithUmap(el, embeddingsList);


  // for multiple
  const baseEl = document.createElement('div');
  const trainedEl = document.createElement('div');
  const movementEl = document.createElement('div');
  [baseEl, trainedEl].forEach(element => {
    element.classList.add('Projector');
    el.appendChild(element);
  });

  // base, trained
  const prng = new Prando(RANDOM_SEED);
  const random = () => prng.next();
  // order of merging matters here
  const projectorOptions = {
    sprites: false,
    color: true,
    ...options,
    umap: {
      random, // fix seed for determinism
      nComponents: 2,
      ...(options.umap || {})
    }, 
  };
  useProjector(baseEl, baseEmbeddingsList, examples, {...projectorOptions, title: 'Embeddings from MobileNet'});
  useProjector(trainedEl, embeddingsList, examples, {...projectorOptions, title: 'Embeddings from your model'});

  // show movement in same (fake) space
  const showMovement = false;
  if (showMovement) {
    const movementEmbeddings = baseEmbeddingsList.concat(embeddingsList);
    const sequences = baseEmbeddingsList.map((embedding, index) => {
      return {
        indices: [index, index + baseEmbeddingsList.length]
      };
    });
    useProjector(movementEl, movementEmbeddings, examples.concat(examples), {...options, sequences});
  }
  
  el.removeChild(waitingEl);
  debug('Done.');
}

async function forEachExample(project, asyncFn) {
  mapExamples(project, async (className, blobUrl, index) => {
    await asyncFn(className, blobUrl, index);
    return undefined;
  });
  return undefined;
}

async function mapExamples(project, asyncFn) {
  const classNames = Object.keys(project.filesByClassName);
  let mapped = [];
  for (var i = 0; i < classNames.length; i++) {
    let className = classNames[i];
    let imageBlobUrls = project.filesByClassName[className] || [];
    for (var j = 0; j < imageBlobUrls.length; j++) {
      let value = await asyncFn(className, imageBlobUrls[j], j);
      mapped.push(value);
    }
  }
  return mapped;
}






async function useProjector(el, embeddingsList, examples, options = {}) {
  // project
  debug('useProjector');
  const umap = new UMAP(options.umap || {});
  debug('  fitting...', options.umap || {});
  const xys = await umap.fitAsync(embeddingsList);

  // reshape for scatterplot
  // metadata is for `showLabelsOnHover`
  const metadata = examples.map((example, i) => {
      const generalizationClassName = examples[i].className;
      const prediction = _.last(_.sortBy(examples[i].predictions, 'probability'));
      const predictedClassName = prediction.className;
      const label = (generalizationClassName === predictedClassName)
        ? `${Math.round(100*prediction.probability)}% ${example.predictions[0].className}`
        : `${Math.round(100*prediction.probability)}% ${example.predictions[0].className} (mislabeled)`;
    return {label};
    // p: example.prediction.probability,
    // label: example.className
    // label: example.prediction.probability
    // label: `${Math.round(100*example.predictions[0].probability)}% ${example.predictions[0].className}`
  });
  const dataset = new ScatterGL.Dataset(xys, metadata);

   // window.xys = xys;
  //  localStorage.setItem('xys', JSON.stringify(xys));
  // console.log('xys', xys);

  debug('  rendering...');
  
  // create spritesheet and attach to dataset
  if (options.sprites) {
    const sprites = examples.map(example => {
      return {uri: example.blobUrl}
    });
    const SPRITE_SHEET_SIZE = 64;
    const spriteSheetImgEl = await createSpriteSheetForScatterplot(sprites, SPRITE_SHEET_SIZE, SPRITE_SHEET_SIZE, {opacity: 0.5});
     console.log('spriteSheetImgEl', spriteSheetImgEl);
    dataset.setSpriteMetadata({
      spriteImage: spriteSheetImgEl,
      singleSpriteSize: [SPRITE_SHEET_SIZE, SPRITE_SHEET_SIZE],
    });
    console.log('spriteMetadata', dataset.spriteMetadata);
  }


  // layout
  const titleEl = document.createElement('h2');
  titleEl.textContent = options.title || 'UMAP projection';
  el.appendChild(titleEl);
  const containerEl = document.createElement('div');
  containerEl.classList.add('Projector-content');
  el.appendChild(containerEl);
  const messageEl = document.createElement('div');
  messageEl.classList.add('Projector-message');
  el.appendChild(messageEl);

  // config
  const scatterGL = new ScatterGL(containerEl, {
    // renderMode: (dataset.spriteMetadata) ? 'SPRITE' : 'POINT',
    onHover: (index) => {
      const d = (index === null ) ? null :{
        example: examples[index],
        xy: xys[index]
      };
      renderHoverMessage(messageEl, d);
    },
    showLabelsOnHover: true, // requires `label` metadata
    selectEnabled: false,
    rotateOnStart: false
    // onSelect: (points) => {
    //   let message = '';
    //   if (points.length === 0 && lastSelectedPoints.length === 0) {
    //     message = '🔥 no selection';
    //   } else if (points.length === 0 && lastSelectedPoints.length > 0) {
    //     message = '🔥 deselected';
    //   } else if (points.length === 1) {
    //     message = `🔥 selected ${points}`;
    //   } else {
    //     message = `🔥selected ${points.length} points`;
    //   }
    //   messageEl.textContent = message;
    // }
  });

  // coloring, tied to number of classes
  if (options.color) {
    // highlight midlines
    scatterGL.setPointColorer(i => {
      // truth, predicted
      // if (i >= examples.length) return '#999'; // grid
      const generalizationClassName = examples[i].className;
      const prediction = _.last(_.sortBy(examples[i].predictions, 'probability'));
      const predictedClassName = prediction.className;
      const hue = (generalizationClassName === predictedClassName) ? 120 : 0;

      return `hsl(${hue}, 100%, ${100 - Math.round(50*prediction.probability)}%)`;
    });

    // alt colors, from tf playground
    // #f59322
    // #e8eaeb
    // #0877bd

    // const labels = _.uniq(examples.map(ex => ex.className)).sort();
    // const CLASSES_COUNT = labels.length*2;
    // const hues = [...new Array(CLASSES_COUNT)].map((_, i) => Math.floor((255 / CLASSES_COUNT) * i));
    // const colorsByLabel = hues.map(hue => `hsl(${hue}, 100%, 30%)`);
    // scatterGL.setPointColorer(i => {
    //   // truth, predicted
    //   const generalizationClassName = examples[i].className;
    //   const predictedClassName = _.last(_.sortBy(examples[i].predictions, 'probability')).className;

    //   const labelIndex = labels.indexOf(generalizationClassName);
    //   const offset = (generalizationClassName === predictedClassName) ? labels.length : 0;
    //   return colorsByLabel[labelIndex + offset];
    // });
  }

  // sequences
  console.log('options.sequences', options.sequences);
  scatterGL.setSequences(options.sequences || []);

  // controls
  scatterGL.setPanMode();

  // dimensions
  scatterGL.setDimensions((options.umap || {}).nComponents || 2);

  // actual render
  scatterGL.render(dataset);
  messageEl.innerHTML = '<div class="Hover" />Hover to see more</div>';

  // seems to have to come after, maybe a bug?
  if (dataset.spriteMetadata) {
    scatterGL.setSpriteRenderMode();
  }

  window.scatterGL = scatterGL;
  window.dataset = dataset;
}





// items is [{uri}]
// img element
async function createSpriteSheetForScatterplot(items, width, height, options = {}) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  context.globalAlpha = options.opacity || 1.0;
  console.log('context.globalAlpha', context.globalAlpha);

  const cols = Math.ceil(Math.sqrt(items.length));
  canvas.width = cols * width;
  canvas.height = cols * width;
  await Promise.all(items.map((item, index) => {
    const x = width * (index % cols);
    const y = height * Math.floor(index / cols);
    const img = new Image();
    return new Promise(function(resolve, reject) {
      img.onload = function() {
        context.drawImage(img, x, y, width, height);
        resolve();
      };
      img.crossOrigin = 'Anonymous';
      img.src = item.uri;
    });
  }));
  
  const uri = canvas.toDataURL();
  const img = document.createElement('img');
  img.width = canvas.width;
  img.height = canvas.height;
  img.src = uri;
  return img;
}

function renderHoverMessage(el, data) {
  const {xys, example} = data;
  const {blobUrl, className, index} = example;
  const id = [className, index].join('-');

  // don't destroy on hover out
  if (data === null) {
    el.style.opacity = 0.5;
  }
  el.innerHTML = `
    <div class="Hover">
      <div><img class="Hover-img" width="224" height="224" /></div>
      <div class="Hover-info">
        <div class="Hover-id"></div>
        <pre class="Hover-debug"></pre>
      </div>
    </div>
  `;
  el.style.opacity = 1.0;
  el.querySelector('.Hover-img').src = blobUrl;
  el.querySelector('.Hover-id').textContent = `id: ${id}`;
  el.querySelector('.Hover-debug').textContent = JSON.stringify(data, null, 2);
  return;
}


function addWaitingEl(el) {
  const waitingEl = document.createElement('div');
  waitingEl.classList.add('Status');
  waitingEl.textContent = 'working...'
  el.appendChild(waitingEl);
  return () => el.removeChild(waitingEl);
}


export async function main() {
  let model = null;
  let closeFn = null;
  let addModelFn = null;

  const webcamStatus = document.querySelector('#webcam-status');
  const webcamButtonEl = document.querySelector('#webcam-start');
  const workspaceEl = document.querySelector('#workspace');
  const modelZipInput = document.querySelector('#model-zip-input');
  const modelStatus = document.querySelector('#model-status');

  webcamButtonEl.addEventListener('click', async e => {
    if (closeFn) {
      closeFn();
      closeFn = null;
      addModelFn = null;
      model = null;
      webcamButtonEl.textContent = 'Start';
      webcamButtonEl.classList.remove('WebcamRunning');
      return;
    }

    closeFn = (await runWebcam(workspaceEl)).close;
    const webcam = await runWebcam(workspaceEl);
    closeFn = webcam.close;
    addModelFn = webcam.addModel;
    webcamButtonEl.textContent = 'Stop';
    webcamButtonEl.classList.add('WebcamRunning');
  });

  modelZipInput.addEventListener('change', async e => {
    if (e.target.files.length === 0) return;
    modelStatus.textContent = 'loading...';
    const [modelZip] = e.target.files;
    model = await loadImageModelFromZipFile(modelZip);
    modelStatus.textContent = 'ready!';
    addModelFn(model);
  });
}

async function runWebcam(el) {
  el.innerHTML = '';
  el.style.margin = '20px';

  console.log('camera-ing...');
  const flipHorizontal = false; // not working as expected
  const webcam = new tmImage.Webcam(224, 224, flipHorizontal);
  await webcam.setup();
  webcam.play();

  // el.appendChild(webcam.webcam);  // bug?
  // el.appendChild(webcam.canvas);  // bug?

   // bug?
  webcam.canvas.width = 224;
  webcam.canvas.height = 224;
  webcam.webcam.width = 224;
  webcam.webcam.height = 224;
  webcam.canvas.style.width = '224px';
  webcam.canvas.style.height = '224px';
  webcam.webcam.style.width = '224px';
  webcam.webcam.style.height = '224px';
  // console.log('webcam.canvas', webcam.canvas);
  // console.log('webcam.webcam', webcam.webcam);
  webcam.webcam.style.margin = '10px';
  webcam.canvas.style.margin = '10px';
  webcam.webcam.style.outline = '10px solid red'; // border impacts sizing!
  webcam.canvas.style.outline = '10px solid orange'; // border impacts sizing!


  // scenes
  // good ones, there are holes in the id space so random doesn't work
  // https://picsum.photos/id/479/224/224
  const goodNumbers = _.shuffle([
    337,
    125,
    69,
    213,
    232,
    110,
    55,
    16,
    289,
    421,
    616,
    479,
    650,
    724,
    564,
    688,
    193,
    458,
    613
  ]);
  const sceneUrls = _.range(0, 9).map(i => {
    // const num = Math.floor(Math.random() * 999);  // id space has holes
    const num = goodNumbers[i];
    return `https://picsum.photos/id/${num}/224/224`
    // imgEl.src = `https://picsum.photos/224/224?${r}`;
    // imgEl.src = 'https://picsum.photos/id/210/224/224';
  });

  // load images
  let sceneImgEls = [];
  for (var i = 0; i < sceneUrls.length; i++) {
    const sceneImgEl = await new Promise((resolve, reject) => {
      const imgEl = document.createElement('img');
      imgEl.onload = () => resolve(imgEl);
      imgEl.onerror = reject;
      imgEl.crossOrigin = 'Anonymous';
      imgEl.src = sceneUrls[i];
    });
    sceneImgEl.width = 224;
    sceneImgEl.height = 224;
    sceneImgEl.style.width = '224px';
    sceneImgEl.style.height = '224px';
    sceneImgEls[i] = sceneImgEl;
  }
  console.log('sceneImgEls', sceneImgEls);

  // realtime
  const realtimeContainerEl = document.createElement('div');
  realtimeContainerEl.classList.add('RealtimeContainer');
  let realtimeEls = [];
  let predictionEls = [];
  let predictionBarEls = [];
  sceneUrls.forEach((url, index) => {
    const el = document.createElement('div');
    el.classList.add('Square');
    realtimeContainerEl.appendChild(el);
    
    // canvas
    const realtimeEl = document.createElement('canvas');
    realtimeEl.classList.add('Realtime');
    realtimeEl.width = 224;
    realtimeEl.height = 224;
    realtimeEl.style.width = '224px';
    realtimeEl.style.height = '224px';
    realtimeEls[index] = realtimeEl;
    el.appendChild(realtimeEl);

    const predictionEl = document.createElement('div');
    predictionEl.classList.add('Prediction');
    el.appendChild(predictionEl);
    predictionEl.textContent = '-';
    predictionEls[index] = predictionEl;
    
    const predictionBarEl = document.createElement('div');
    predictionBarEl.classList.add('PredictionBar');
    el.appendChild(predictionBarEl);
    predictionBarEls[index] = predictionBarEl;
  });
  el.appendChild(realtimeContainerEl);

  // click for new scene
  realtimeEls.forEach((realtimeEl, i) => {
    realtimeEl.addEventListener('click', e => {
      console.log('click!');
      const num = Math.floor(Math.random() * 299);  // id space has holes
      sceneUrls[i] = `https://picsum.photos/id/${num}/224/224`;
      sceneImgEls[i].src = sceneUrls[i];
    });
  });

  console.log('loading...');
  const net = await bodyPix.load({
    // architecture: 'ResNet50',
    // outputStride: 32,
    // quantBytes: 2
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2
  });
  console.log('loaded.');

  // local state for the core loop
  let shouldAbort = false;
  let model = null;
  let ticker = 0;

  async function tick() {
    if (shouldAbort) {
      console.log('aborted.');
      return;
    }

    // scaling bug?
    // console.log('webcam.webcam.videoWidth', webcam.webcam.videoWidth)
    // console.log('webcam.webcam.videoHeight', webcam.webcam.videoHeight);

    webcam.update();
    const croppedEl = cropTo(webcam.canvas, 224, flipHorizontal);
    // console.log('segmenting...');
    const outputStride = 16;
    const segmentationThreshold = 0.70; // over default 0.5
    // const segmentation = await net.segmentPerson(croppedEl, outputStride, segmentationThreshold);
    const segmentation = await net.segmentPerson(croppedEl, {
      flipHorizontal,
      internalResolution: 'high',
      segmentationThreshold: 0.6
    });

    // console.log('computing masks...');
    const imageMask = bodyPix.toMask(segmentation);
    const sceneMask = bodyPix.toMask(segmentation, {r: 0, g: 0, b: 0, a: 255}, {r: 0, g: 0, b: 0, a: 0}); // invert

    for (var i = 0; i < sceneUrls.length; i++) {
      const compositedEl = composite(croppedEl, imageMask, sceneImgEls[i], sceneMask);
      redraw(compositedEl, realtimeEls[i]);

      // sample, since it's too slow.  predict one each tick
      if (model && (ticker % sceneUrls.length) === i) {
        const predictions = await model.predict(compositedEl);
        const prediction = _.last(_.sortBy(predictions, 'probability'));
        predictionEls[i].textContent = `${Math.round(100*prediction.probability)}% ${prediction.className}`;
        
        const colors = ['blue', 'orange', 'purple', 'brown'];
        const colorIndex = predictions.map(p => p.className).indexOf(prediction.className);
        predictionBarEls[i].style.background = colors[colorIndex];
        predictionBarEls[i].style.width = `${Math.round(224*prediction.probability)}px`;

        // const fn = await createBarGraph(predictionEl, labels, predictions);
      }
    }

    ticker = ticker + 1;
    setTimeout(tick, 16);//requestAnimationFrame(tick);
  }

  tick();

  function close() {
    webcam.stop();
    shouldAbort = true;
  }

  function addModel(passedModel) {
    console.log('addModel', passedModel);
    model = passedModel;
  }

  return {close, addModel};
}


function redraw(input, output) {
  const originalImageFrame = input.getContext('2d').getImageData(0, 0, 224, 224);
  output.getContext('2d').putImageData(originalImageFrame, 0, 0);
  return;
}

function composite(imageCanvas, imageMask, sceneImageEl, sceneMask) {
  const opacity = 1.0;
  const maskBlurAmount = 0.1;
  const pixelCellWidth = 1;
  
  // console.log('drawing image mask...');
  const maskedImageCanvas = document.createElement('canvas');
  maskedImageCanvas.width = 224;
  maskedImageCanvas.height = 224;
  maskedImageCanvas.style.width = '224px';
  maskedImageCanvas.style.height = '224px';
  maskedImageCanvas.classList.add('Masked');
  bodyPix.drawMask(maskedImageCanvas, imageCanvas, imageMask, opacity, maskBlurAmount, false, pixelCellWidth);

  // console.log('drawing scene mask...');
  const maskedSceneCanvas = document.createElement('canvas');
  bodyPix.drawMask(maskedSceneCanvas, sceneImageEl, sceneMask, opacity, maskBlurAmount, false, pixelCellWidth);

  // composite
  // console.log('composite...');
  const composited = document.createElement('canvas');
  composited.width = 224;
  composited.height = 224;

  // low budget compositing
  const ctx = composited.getContext('2d');
  const outFrame = ctx.createImageData(224, 224);
  const imgFrame = maskedImageCanvas.getContext('2d').getImageData(0, 0, 224, 224);
  const sceneFrame = maskedSceneCanvas.getContext('2d').getImageData(0, 0, 224, 224);
  let l = imgFrame.data.length / 4;
  for (let i = 0; i < l; i++) {
    let r = imgFrame.data[i * 4 + 0];
    let g = imgFrame.data[i * 4 + 1];
    let b = imgFrame.data[i * 4 + 2];
    let a = imgFrame.data[i * 4 + 3];
    if (r === 0 && g === 0 && b === 0) {;
      outFrame.data[i * 4 + 0] = sceneFrame.data[i * 4 + 0];
      outFrame.data[i * 4 + 1] = sceneFrame.data[i * 4 + 1];
      outFrame.data[i * 4 + 2] = sceneFrame.data[i * 4 + 2];
      outFrame.data[i * 4 + 3] = sceneFrame.data[i * 4 + 3];
    } else {
      outFrame.data[i * 4 + 0] = imgFrame.data[i * 4 + 0];
      outFrame.data[i * 4 + 1] = imgFrame.data[i * 4 + 1];
      outFrame.data[i * 4 + 2] = imgFrame.data[i * 4 + 2];
      outFrame.data[i * 4 + 3] = imgFrame.data[i * 4 + 3];
    }
  }
  ctx.putImageData(outFrame, 0, 0);
  return composited;
}
