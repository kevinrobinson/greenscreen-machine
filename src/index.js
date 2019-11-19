// console.log('dependencies', {JSZip, _, tmImage, UMAP, ScatterGL});

const RANDOM_SEED = 42;
console.log('RANDOM_SEED', RANDOM_SEED);

function debug(...params) {
  console.log(...params); // eslint-disable-line no-console
}

async function setImageSrc(imgEl, src) {
  await new Promise((resolve, reject) => {
    imgEl.onload = resolve;
    imgEl.onerror = reject;
    imgEl.src = src;
  });
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


/*
export declare interface TeachableMachineImageProject {
  manifest: TeachableMachineImageProjectManifest
  filesByClassName: {[className: string]: [blobUrl]}
}
*/
async function loadImageProjectFromZipFile(projectZipFile) {
  debug('Opening project zip...');
  const jsZip = new JSZip();
  const zip = await jsZip.loadAsync(projectZipFile);

  debug('Reading manifest.json...')
  let manifestEntry = null;
  zip.forEach((relativePath, entry) => {
    if (relativePath === 'manifest.json' && entry.name === 'manifest.json') {
      manifestEntry = entry;
    }
  });
  const manifest = (manifestEntry === null)
    ? null
    : JSON.parse(await manifestEntry.async('text'));
  debug(manifest ? '  manifest found' : '  manifest not found');

  // use a for loop for simple async/await usage
  debug('Reading image files...');
  const filesByClassName = {};
  for (var i = 0; i < Object.keys(zip.files).length; i++) {
    const relativePath = Object.keys(zip.files)[i];
    if (relativePath === 'manifest.json') continue;

    const [className, exampleNumber] = relativePath.split('-!-');
    filesByClassName[className] || (filesByClassName[className] = []);
    if (filesByClassName[className][exampleNumber] !== undefined) {
      console.warn('unexpected project file format');
    }

    const entry = zip.files[relativePath];
    const blob = await entry.async('blob');
    const blobUrl = URL.createObjectURL(blob);
    filesByClassName[className].push(blobUrl);
  };

  console.log('Done.');
  return {manifest, filesByClassName};
}

async function inspect(generalization, model, inspectorEl, deps) {
  const {createBarGraph} = deps;
  const project = generalization; // test against generalization

  // labels from model, classNames from project dataset
  inspectorEl.innerHTML = '';
  const labels = model.getClassLabels();
  const classNames = Object.keys(project.filesByClassName);
  const sameLabelsAcrossDatasets = _.isEqual(labels.sort(), classNames.sort());
  await Promise.all(classNames.map(async className => {
    // for each class in model
    console.log('  inspect, className:', className);
    const classEl = document.createElement('div');
    classEl.classList.add('InspectClass');
    const titleEl = document.createElement('h2');
    titleEl.classList.add('InspectClass-title');
    titleEl.textContent = `Generalization label: ${className}`;
    classEl.appendChild(titleEl);

    // add all images and the model prediction for that image
    const imageBlobUrls = project.filesByClassName[className] || [];
    console.log('    imageBlobUrls.length', imageBlobUrls.length);
    await Promise.all(imageBlobUrls.map(async (blobUrl, index) => {
      const exampleEl = document.createElement('div');
      exampleEl.classList.add('InspectExample');

      console.log('    index:', index);
      const imgEl = document.createElement('img');
      imgEl.classList.add('InspectExample-img');
      imgEl.title = index;
      exampleEl.appendChild(imgEl);
      await setImageSrc(imgEl, blobUrl); // before predicting

      const labelEl = document.createElement('div');
      labelEl.classList.add('InspectExample-label');
      labelEl.textContent = className;
      exampleEl.appendChild(labelEl);

      const predictionEl = document.createElement('div');
      predictionEl.classList.add('InspectExample-prediction');
      predictionEl.classList.add('graph-wrapper');
      const predictions = await model.predict(imgEl);
      // const prediction = _.last(_.sortBy(predictions, 'probability'));
      // predictionEl.textContent = `model says: ${prediction.className}, ${Math.round(100*prediction.probability)}%`;
      const fn = await createBarGraph(predictionEl, labels, predictions);
      exampleEl.appendChild(predictionEl);

      // only highlight if model labels and dataset labels match
      if (sameLabelsAcrossDatasets) {
        const prediction = _.last(_.sortBy(predictions, 'probability'));
        if (className === prediction.className) {
          exampleEl.classList.add('InspectExample-prediction-does-match');
        } else {
          exampleEl.classList.add('InspectExample-prediction-does-not-match');
        }
      }
      classEl.appendChild(exampleEl);
    }));

    inspectorEl.appendChild(classEl);
  }));
}



// This is dependent on how the TM image model
// is constructed by the training process.  It's
// two layers - a truncated MobileNet to get embeddings,
// with a smaller trained model on top.  That trained
// model has two layers itself - a dense layer and a softmax
// layer.
//
// So we get the trained model first, then within
// there we apply the second-to-last layer to get
// embeddings.
//
// In other words, take the last sofmax layer off
// the last layer.
function infer(tmImageModel, raster) {
  const tfModel = tmImageModel.model;
  const seq = tf.sequential();
  seq.add(_.first(tfModel.layers)); // mobilenet
  seq.add(_.first(_.last(tfModel.layers).layers)); // dense layer, without softmax
  return seq.predict(capture(raster));
}

function inferMobileNet(tmImageModel, raster) {
  const tfModel = tmImageModel.model;
  const seq = tf.sequential();
  seq.add(_.first(tfModel.layers)); // mobilenet embeddings only
  return seq.predict(capture(raster));
}

// doesn't work, just grabbing dense layer already has inbound
// connections from previous
function inferFromMobileNetEmbedding(tmImageModel, mobileNetEmbedding) {
  const tfModel = tmImageModel.model;

  // try to just rewire
  // const denseLayer = _.first(_.last(tfModel.layers).layers);
  // denseLayer.inboundNodes = [];
  // const seq2 = tf.sequential();
  // seq2.add(denseLayer); // mobilenet embeddings only
  // return seq.predict(mobileNetEmbedding);

  // try to rebuild from config and weights
  const denseLayer = _.first(_.last(tfModel.layers).layers);
  const rewiredDenseLayer = tf.layers.dense({
    ...denseLayer.getConfig(),
    inputShape: [null, 1280]
  });
  // rewiredDenseLayer.build();
  // doesn't work
  // rewiredDenseLayer.setWeights(denseLayer.getWeights());
  const seq = tf.sequential({
    layers: [rewiredDenseLayer]
  });
  return seq.predict(mobileNetEmbedding);
}

// copied
function capture(rasterElement) {
    return tf.tidy(() => {
        const pixels = tf.browser.fromPixels(rasterElement);

        // crop the image so we're using the center square
        const cropped = cropTensor(pixels);

        // Expand the outer most dimension so we have a batch size of 1
        const batchedImage = cropped.expandDims(0);

        // Normalize the image between -1 and a1. The image comes in between 0-255
        // so we divide by 127 and subtract 1.
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

// copied
function cropTensor(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
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




// needs more than n=15 by default
// async function projectWithUmap(el, embeddingsList) {
//   console.log('projectWithUmap', embeddingsList.length);
//   const umap = new UMAP();
//   console.log('fitting', umap);
//   const xys = await umap.fitAsync(embeddingsList);
//   console.log('xys', xys);
//   const xDomain = [_.min(xys.map(xy => xy[0])), _.max(xys.map(xy => xy[0]))];
//   const yDomain = [_.min(xys.map(xy => xy[1])), _.max(xys.map(xy => xy[1]))];
//   console.log('xDomain', xDomain);
//   console.log('yDomain', yDomain);
  
//   var xScale = d3.scaleLinear()
//       .domain(xDomain)
//       .range([ 0, 800 ]);
//   var yScale = d3.scaleLinear()
//       .domain(yDomain)
//       .range([ 0, 600 ]);
//   const ns = "http://www.w3.org/2000/svg";
//   const svg = document.createElementNS(ns, 'svg');
//   svg.setAttribute('width', 800);
//   svg.setAttribute('height', 600);
//   svg.style.width = '800px';
//   svg.style.height = '600px';
  
//   console.log('projected', xys.map(xy => [xScale(xy[0]), yScale(xy[1])]));
//   xys.forEach((xy, index) => {
//     const [x, y] = xy;
//     const circle = document.createElementNS(ns, 'circle');
//     circle.setAttribute('cx', xScale(x));
//     circle.setAttribute('cy', yScale(y));
//     circle.setAttribute('r', 5);
//     const i = Math.round(index / xys.length * 16);
//     circle.setAttribute('fill', `#ff${i.toString(16)}`); // rgb didn't work, even in web inspector? confused, but working around...
//     svg.appendChild(circle);
//   });
//   el.appendChild(svg);
// }


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
  let closeFn = null;
  const buttonEl = document.querySelector('.Button');
  const instructionsEl = document.querySelector('.IntructionsText');
  const workspaceEl = document.querySelector('#workspace');
  buttonEl.addEventListener('click', async e => {
    if (closeFn) {
      closeFn();
      closeFn = null;
      buttonEl.textContent = 'Start';
      instructionsEl.style.display = 'block';
      return;
    }

    closeFn = (await runWebcam(workspaceEl)).close;
    buttonEl.textContent = 'Stop';
    instructionsEl.style.display = 'none';
  });
}

async function runWebcam(el) {
  el.innerHTML = '';
  el.style.margin = '20px';

  console.log('camera-ing...');

  // const video = document.createElement('video');
  // const canvas = document.createElement('canvas');
  // el.appendChild(video);
  // el.appendChild(canvas);
  // canvas.width = 224;
  // canvas.height = 224;
  // video.width = 224;
  // video.height = 224;
  // video.autoPlay = 'true';
  // const videoOptions = {};
  // await new Promise((resolve, reject) => {
  //   window.navigator.mediaDevices.getUserMedia({ video: videoOptions }).then(mediaStream => {
  //     video.srcObject = mediaStream;
  //     video.addEventListener('loadedmetadata', event => {
  //         const { videoWidth: vw, videoHeight: vh } = video;
  //         video.width = vw;
  //         video.height = vh;
  //         resolve();
  //     });
  //   }, () => {
  //     reject('Could not open your camera. You may have denied access.');
  //   });
  // });
        
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
  console.log('webcam.canvas', webcam.canvas);
  console.log('webcam.webcam', webcam.webcam);
  webcam.webcam.style.margin = '10px';
  webcam.canvas.style.margin = '10px';
  webcam.webcam.style.outline = '10px solid red'; // border impacts sizing!
  webcam.canvas.style.outline = '10px solid orange'; // border impacts sizing!


  // scenes
  // good ones:
  // https://picsum.photos/id/479/224/224
  const goodNumbers = _.shuffle([337, 289, 421, 616, 479, 650, 724, 564, 688, 193, 458, 613]);
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
  const realtimeEls = sceneUrls.map((url, index) => {
    const realtimeEl = document.createElement('canvas');
    realtimeEl.classList.add('Realtime');
    realtimeEl.width = 224;
    realtimeEl.height = 224;
    realtimeEl.style.width = '224px';
    realtimeEl.style.height = '224px';
    realtimeContainerEl.appendChild(realtimeEl);
    return realtimeEl;
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

  let shouldAbort = false;
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
    console.log('segmenting...');
    const outputStride = 16;
    const segmentationThreshold = 0.70; // over default 0.5
    // const segmentation = await net.segmentPerson(croppedEl, outputStride, segmentationThreshold);
    const segmentation = await net.segmentPerson(croppedEl, {
      flipHorizontal,
      internalResolution: 'high',
      segmentationThreshold: 0.6
    });

    console.log('computing masks...');
    const imageMask = bodyPix.toMask(segmentation);
    const sceneMask = bodyPix.toMask(segmentation, {r: 0, g: 0, b: 0, a: 255}, {r: 0, g: 0, b: 0, a: 0}); // invert

    for (var i = 0; i < sceneUrls.length; i++) {
      const compositedEl = composite(croppedEl, imageMask, sceneImgEls[i], sceneMask);
      redraw(compositedEl, realtimeEls[i]);
    }

    setTimeout(tick, 16);//requestAnimationFrame(tick);
  }

  tick();

  function close() {
    webcam.stop();
    shouldAbort = true;
  }

  return {close};
}


// async function runBodyPix(project, el) {
//   console.log('loading bodypix...');
//   const net = await bodyPix.load();

//   // const base = 'ade20k';
//   // const quantizationBytes = 2;
//   // const deepLab = await deeplab.load({base, quantizationBytes});
//   // const colormap = deeplab.getColormap(base);
//   // const labels = deeplab.getLabels(base);
//   // console.log('colormap', colormap);
//   // console.log('labels', labels);

//   console.log('segmenting...', net);
//   const uris = await mapExamples(project, async (className, blobUrl, index) => blobUrl);
//   console.log('uris', uris);
//   const n = 1;
//   for (var i = 0; i < n; i++) {
//     await Promise.all(uris.map(async (blobUrl, index) => {
//       const containerEl = document.createElement('div');
//       // containerEl.style.display = 'flex';
//       containerEl.style.display = 'inline-block';
//       containerEl.style.marging = '10px';
//       // containerEl.style['flex-direction'] = 'row';
//       el.appendChild(containerEl);

//       const imgEl = await imageFromUri(blobUrl);
//       imgEl.width = 224;
//       imgEl.height = 224;
//       // containerEl.appendChild(imgEl);

//       // const output = await deepLab.segment(imgEl);
//       // console.log('  output', output);
//       // const {height, width, segmentationMap} = output;
//       // const segmentationPixels = new ImageData(segmentationMap, width, height);
//       // console.log('  segmentationPixels', segmentationPixels);
//       // const overlayEl = canvasOverlay(imgEl, segmentationPixels);

//       console.log('  index', index);
//       await make(net, containerEl, imgEl);
//     }));
//   }
// }

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

function canvasOverlay(imgEl, segmentationPixels) {

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.classList.add('Image-overlay');
  canvas.classList.add('Image-overlay-canvas');
  canvas.style.width = getComputedStyle(imgEl).width;
  canvas.style.height = getComputedStyle(imgEl).height;
  canvas.width = imgEl.width;
  canvas.height = imgEl.height;
  ctx.putImageData(segmentationPixels, 0, 0);
  return canvas;
}


async function imageFromUri(src) {
  const imgEl = document.createElement('img');
  await new Promise((resolve, reject) => {
    imgEl.onload = resolve;
    imgEl.onerror = reject;
    imgEl.src = src;
  });
  return imgEl;
}




// async function make(net, el, canvasEl, sceneUrl) {
//   // const overlayEl = canvasOverlay(rasterEl, maskImagePixels);

//   console.log('container...');
//   const containerEl = document.createElement('div');
//   containerEl.style.display = 'inline-block';
//   containerEl.style.margin = '10px';
//   containerEl.style.padding = '10px';
//   containerEl.style.border = '1px solid red';
//   // el.appendChild(containerEl);
//   console.log('containerEl', containerEl);

//   // console.log('copied...');
//   // const copied = document.createElement('canvas');
//   // copied.classList.add('Copied');
//   // copied.width = 224;
//   // copied.height = 224;
//   // copied.style.width = '224px';
//   // copied.style.height = '224px';
//   // const originalImageFrame = canvasEl.getContext('2d').getImageData(0, 0, 224, 224);
//   // copied.getContext('2d').putImageData(originalImageFrame, 0, 0);
//   // containerEl.appendChild(copied);
//   // console.log('copied', copied);

//   // console.log('segmenting...');
//   // const segmentation = await net.segmentPerson(canvasEl); //, outputStride, segmentationThreshold);
//   // console.log('segmentation', segmentation);

//   // console.log('computing mask...');
//   // const mask = bodyPix.toMask(segmentation);

//   console.log('drawing mask...');
//   const canvas = document.createElement('canvas');
//   canvas.width = 224;
//   canvas.height = 224;
//   canvas.style.width = '224px';
//   canvas.style.height = '224px';
//   canvas.classList.add('Masked');
//   containerEl.appendChild(canvas);
//   const maskBlurAmount = 0.1;
//   const pixelCellWidth = 1;
//   const opacity = 1.0;
//   console.log('  els', canvas, canvasEl, mask);
//   console.log('  dims', [canvas.width, canvas.height], [canvasEl.width, canvasEl.height]);
//   bodyPix.drawMask(canvas, canvasEl, mask, opacity, maskBlurAmount, false, pixelCellWidth);
//   // console.log('  overlayEl', overlayEl);

//   console.log('scene...', sceneUrl);
//   const sceneImgEl = await new Promise((resolve, reject) => {
//     const imgEl = document.createElement('img');
//     imgEl.onload = () => resolve(imgEl);
//     imgEl.onerror = reject;
//     imgEl.crossOrigin = 'Anonymous';
//     imgEl.src = sceneUrl;
//   });
//   sceneImgEl.width = 224;
//   sceneImgEl.height = 224;

//   containerEl.appendChild(sceneImgEl);

//   // return composite(segmentation, sceneImgEl, canvas);
//   console.log('scene mask...');
//   const sceneCanvas = document.createElement('canvas');
//   const sceneMask = bodyPix.toMask(segmentation, {r: 0, g: 0, b: 0, a: 255}, {r: 0, g: 0, b: 0, a: 0}); // invert
//   bodyPix.drawMask(sceneCanvas, sceneImgEl, sceneMask, opacity, maskBlurAmount, false, pixelCellWidth);
//   containerEl.appendChild(sceneCanvas);

//   // composite
//   console.log('composite...');
//   const composited = document.createElement('canvas');
//   composited.width = 224;
//   composited.height = 224;

//   // low budget compositing
//   const ctx = composited.getContext('2d');
//   const outFrame = ctx.createImageData(224, 224);
//   const imgFrame = canvas.getContext('2d').getImageData(0, 0, 224, 224);
//   const sceneFrame = sceneCanvas.getContext('2d').getImageData(0, 0, 224, 224);
//   let l = imgFrame.data.length / 4;
//   for (let i = 0; i < l; i++) {
//     let r = imgFrame.data[i * 4 + 0];
//     let g = imgFrame.data[i * 4 + 1];
//     let b = imgFrame.data[i * 4 + 2];
//     let a = imgFrame.data[i * 4 + 3];
//     if (r === 0 && g === 0 && b === 0) {;
//       outFrame.data[i * 4 + 0] = sceneFrame.data[i * 4 + 0];
//       outFrame.data[i * 4 + 1] = sceneFrame.data[i * 4 + 1];
//       outFrame.data[i * 4 + 2] = sceneFrame.data[i * 4 + 2];
//       outFrame.data[i * 4 + 3] = sceneFrame.data[i * 4 + 3];
//     } else {
//       outFrame.data[i * 4 + 0] = imgFrame.data[i * 4 + 0];
//       outFrame.data[i * 4 + 1] = imgFrame.data[i * 4 + 1];
//       outFrame.data[i * 4 + 2] = imgFrame.data[i * 4 + 2];
//       outFrame.data[i * 4 + 3] = imgFrame.data[i * 4 + 3];
//     }
//   }
//   ctx.putImageData(outFrame, 0, 0);
//   containerEl.appendChild(composited);
//   return composited;
// }
