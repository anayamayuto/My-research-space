/**
 * lecture-client.js
 * A-Frame VR講義空間 + MediaPipe Pose Landmarker による姿勢トラッキング
 */

// --- Imports (ESM via CDN) ---
import {
  PoseLandmarker,
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ========== State ==========
const state = {
  poseLandmarker: null,
  faceLandmarker: null,
  webcamRunning: false,
  videoLoaded: false,
  sessionId: crypto.randomUUID(),
  lastSendTime: 0,
  sendInterval: 5000, // 5s
  scores: { shoulderTilt: 0, headForward: 0, bodyLean: 0, overall: 0 },
  gaze: {
    cameraRotation: { x: 0, y: 0, z: 0 },
    headDirection: { yaw: 0, pitch: 0, roll: 0 },
    lookingAt: "other",
    gazeStability: 0,
  },
  engagement: {
    isLookingAtScreen: false,
    postureQuality: "bad",
    secondsSinceLastMovement: 0,
  },
  emotion: { frown: 0, smile: 0, eyeOpenness: 1, jawOpen: 0 },
  lastProbs: null, // latest 5-class probabilities (for the live monitor)
  audio: { volumeLevel: 0 },
  interaction: { lastAction: "none", timeSinceAction: 0, _lastActionTime: Date.now() },
  context: { currentText: "", _subtitles: [] },
  hands: { handsVisible: 0, isTouchingFace: false },
  aiTriggerTracker: {
    confusionStart: null, distractionStart: null, badPostureStart: null, flowStart: null,
    lastMessageTime: 0, currentPattern: null
  },
  dataBuffer: [],
  videoStartTime: null,
  // Gaze stability tracking
  _prevCamRot: { x: 0, y: 0 },
  _lastMovementTime: Date.now(),
  _gazeHistory: [], // recent lookingAt values for stability
  mlModel: null,
  mlScaler: null,
  annotationLabel: null,
  annotationBuffer: [],
  // ML prediction smoothing
  mlPredHistory: [],   // recent {classId, prob} predictions
  _lastPredictTime: 0, // throttle inference
};

// ===== ML inference constants =====
const ML_PREDICT_INTERVAL = 200; // run inference at most every 200ms (5fps)
const ML_PROB_THRESHOLD = 0.7;   // ignore low-confidence predictions
const ML_HISTORY_SIZE = 10;      // majority vote over last 10 predictions (~2s)
const ML_MAJORITY_MIN = 6;       // need 6/10 agreement to accept a class
const ML_Z_CLIP = 3;             // clip standardized features to [-3, 3] to avoid
                                 // wild extrapolation outside the training range

/**
 * Build the ML feature vector (11 features).
 * MUST stay in sync with ml/preprocess.py and server.js CSV header:
 * [overall, shoulderTilt, bodyLean, yaw, pitch, rollSin, rollCos,
 *  frown, eyeOpenness, handsVisible, isTouchingFace]
 * Notes:
 * - headForward was removed (constant 0 in training data → scaler blow-up)
 * - smile was removed (near-constant → noise amplification)
 * - roll is encoded as sin/cos because it wraps around ±180°
 * - eyeOpenness uses ?? so that a legit value of 0 (eyes closed) is kept
 */
function buildFeatureVector() {
  const s = state.scores;
  const g = state.gaze;
  const emo = state.emotion;
  const hnds = state.hands;
  const rollDeg = g.headDirection?.roll ?? 0;
  const rollRad = (rollDeg * Math.PI) / 180;
  return [
    s.overall || 0,
    s.shoulderTilt || 0,
    s.bodyLean || 0,
    g.headDirection?.yaw ?? 0,
    g.headDirection?.pitch ?? 0,
    Math.sin(rollRad),
    Math.cos(rollRad),
    emo.frown ?? 0,
    emo.eyeOpenness ?? 1,
    hnds.handsVisible || 0,
    hnds.isTouchingFace ? 1 : 0,
  ];
}

// Audio & Worker variables
let audioContext = null;
let audioAnalyser = null;
let audioDataArray = null;
let trackingWorker = null;

// ========== DOM Elements ==========
const $ = (s) => document.getElementById(s);
const videoInput = $("videoInput");
const vttInput = $("vttInput");
const loadBtn = $("loadBtn");
const pauseBtn = $("pauseBtn");
const startPoseBtn = $("startPoseBtn");
const stopPoseBtn = $("stopPoseBtn");
const mainVideo = $("mainVideo");
const webcamVideo = $("webcamVideo");
const poseCanvas = $("poseCanvas");
const poseCtx = poseCanvas.getContext("2d");
const webcamPip = $("webcam-pip");
const posturePanel = $("posture-panel");
const aiSupporterPanel = $("ai-supporter-panel");
const aiMessageText = $("ai-message-text");
const aiActionsContainer = $("ai-actions-container");
const statusDot = $("statusDot");
const loadingOverlay = $("loading-overlay");

// Score elements
const overallScoreNum = $("overallScoreNum");
const overallProgress = $("overallProgress");
const shoulderVal = $("shoulderVal");
const shoulderBar = $("shoulderBar");
const headVal = $("headVal");
const headBar = $("headBar");
const leanVal = $("leanVal");
const leanBar = $("leanBar");

// Status bar elements
const statusTracking = $("statusTracking");
const statusSession = $("statusSession");
const statusDataPoints = $("statusDataPoints");
const statusGaze = $("statusGaze");

// ========== Toast ==========
function showToast(msg, duration = 3000) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// ========== MediaPipe Initialization ==========
async function initPoseLandmarker() {
  showToast("🧠 AIモデルを読み込み中...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  
  // Pose Landmarker
  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  // Face Landmarker (for emotion)
  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });

  // Hand Landmarker (for gestures)
  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  showToast("✅ AIモデル準備完了");
}

// ========== Webcam & Audio ==========
async function startWebcam() {
  if (state.webcamRunning) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: true // Request audio for volume detection
    });
    webcamVideo.srcObject = stream;
    await webcamVideo.play();
    poseCanvas.width = webcamVideo.videoWidth;
    poseCanvas.height = webcamVideo.videoHeight;
    state.webcamRunning = true;
    resetDrowsyTracking();
    webcamPip.classList.remove("hidden");
    posturePanel.classList.remove("hidden");
    aiSupporterPanel.classList.remove("hidden");
    statusDot.classList.add("active");
    startPoseBtn.style.display = "none";
    stopPoseBtn.style.display = "inline-flex";
    statusTracking.textContent = "トラッキング中";
    
    // Setup Audio Context for volume
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      audioAnalyser = audioContext.createAnalyser();
      audioAnalyser.fftSize = 256;
      source.connect(audioAnalyser);
      audioDataArray = new Uint8Array(audioAnalyser.frequencyBinCount);
    } catch (err) {
      console.warn("Audio context init failed:", err);
    }

    showToast("📷 カメラ・マイク起動・トラッキング開始");
    
    // Create a Web Worker from a Blob to act as a background timer (approx 30fps)
    const workerCode = `
      let interval;
      self.onmessage = function(e) {
        if (e.data === 'start') interval = setInterval(() => postMessage('tick'), 33);
        if (e.data === 'stop') clearInterval(interval);
      };
    `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    trackingWorker = new Worker(URL.createObjectURL(blob));
    trackingWorker.onmessage = () => detectPose();
    trackingWorker.postMessage('start');
    
  } catch (e) {
    console.error("Webcam error:", e);
    showToast("⚠️ カメラ・マイクへのアクセスに失敗しました");
  }
}

function stopWebcam() {
  state.webcamRunning = false;
  if (webcamVideo.srcObject) {
    webcamVideo.srcObject.getTracks().forEach((t) => t.stop());
    webcamVideo.srcObject = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
  }
  if (trackingWorker) {
    trackingWorker.postMessage('stop');
    trackingWorker.terminate();
    trackingWorker = null;
  }
  webcamPip.classList.add("hidden");
  posturePanel.classList.add("hidden");
  statusDot.classList.remove("active");
  startPoseBtn.style.display = "inline-flex";
  stopPoseBtn.style.display = "none";
  statusTracking.textContent = "停止中";
  // Send remaining buffer
  if (state.dataBuffer.length > 0) sendPoseData();
  showToast("⏹ トラッキング停止");
}

// ========== Pose Detection Loop ==========
let lastDetectTime = 0;
let lastCaptureTime = 0; // Throttle data buffering to 5fps

function detectPose() {
  if (!state.webcamRunning || !state.poseLandmarker) return;

  // Ensure webcam video is playing and has valid dimensions
  if (webcamVideo.readyState < 2 || webcamVideo.videoWidth === 0) {
    return;
  }

  const now = performance.now();
  // MediaPipe requires strictly increasing timestamps
  if (now <= lastDetectTime) {
    return;
  }
  lastDetectTime = now;

  try {
    let result = null;
    try {
      result = state.poseLandmarker.detectForVideo(webcamVideo, now);
    } catch (e) {
      // Ignored if pose landmarker fails on a frame
    }
    
    let faceResult = null;
    if (state.faceLandmarker) {
      try {
        faceResult = state.faceLandmarker.detectForVideo(webcamVideo, now);
      } catch (e) {
        // Ignored
      }
    }

    let handResult = null;
    if (state.handLandmarker) {
      try {
        handResult = state.handLandmarker.detectForVideo(webcamVideo, now);
      } catch (e) {
        // Ignored
      }
    }

    poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

    // Update audio volume
    if (audioAnalyser && audioDataArray) {
      audioAnalyser.getByteFrequencyData(audioDataArray);
      let sum = 0;
      for (let i = 0; i < audioDataArray.length; i++) {
        sum += audioDataArray[i];
      }
      const avg = sum / audioDataArray.length;
      state.audio.volumeLevel = Math.round((avg / 255) * 100);
    }

    // Process Face features
    if (faceResult && faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
      const shapes = faceResult.faceBlendshapes[0].categories;
      let browDownL = 0, browDownR = 0;
      let smileL = 0, smileR = 0;
      let blinkL = 0, blinkR = 0;
      let jawOpen = 0;

      for (const s of shapes) {
        if (s.categoryName === "browDownLeft") browDownL = s.score;
        if (s.categoryName === "browDownRight") browDownR = s.score;
        if (s.categoryName === "mouthSmileLeft") smileL = s.score;
        if (s.categoryName === "mouthSmileRight") smileR = s.score;
        if (s.categoryName === "eyeBlinkLeft") blinkL = s.score;
        if (s.categoryName === "eyeBlinkRight") blinkR = s.score;
        if (s.categoryName === "jawOpen") jawOpen = s.score;
      }

      state.emotion.frown = Math.round(Math.max(browDownL, browDownR) * 100) / 100;
      state.emotion.smile = Math.round(Math.max(smileL, smileR) * 100) / 100;
      const avgBlink = (blinkL + blinkR) / 2;
      state.emotion.eyeOpenness = Math.round((1.0 - avgBlink) * 100) / 100;
      state.emotion.jawOpen = Math.round(jawOpen * 100) / 100;
    }

    // Process Pose
    let hasPose = false;
    let nose = null, leye = null, reye = null;
    if (result && result.landmarks && result.landmarks.length > 0) {
      const lm = result.landmarks[0];
      drawSkeleton(lm);
      calculateScores(lm);
      calculateHeadDirection(lm);
      nose = lm[0]; leye = lm[2]; reye = lm[5];
      hasPose = true;
    }

    // Process Hands (Face touching detection)
    state.hands.handsVisible = 0;
    state.hands.isTouchingFace = false;
    if (handResult && handResult.landmarks && handResult.landmarks.length > 0) {
      state.hands.handsVisible = handResult.landmarks.length;
      
      // Check distance between hands and face if face is detected
      if (nose) {
        for (const hand of handResult.landmarks) {
          const indexFinger = hand[8]; // index finger tip
          const wrist = hand[0];
          
          // Distance squared calculation (rough 2D heuristic)
          const distToIndex = Math.pow(indexFinger.x - nose.x, 2) + Math.pow(indexFinger.y - nose.y, 2);
          const distToWrist = Math.pow(wrist.x - nose.x, 2) + Math.pow(wrist.y - nose.y, 2);
          
          // Relaxed distance so it's easier to detect face touches
          if (distToIndex < 0.04 || distToWrist < 0.05) {
            state.hands.isTouchingFace = true;
            break;
          }
        }
      }
    }

    updateGazeFromCamera();
    calculateEngagement();
    updateScoreUI();
    drowsyRealtimeUpdate(Date.now()); // per-frame blink/yawn state machines
    checkAITriggers();

    // Limit data capture to ~5 times per second (200ms) to avoid massive JSON sizes
    if (Date.now() - lastCaptureTime > 200) {
      lastCaptureTime = Date.now();
      
      // Buffer data point with full AI training structure
      const videoTime = mainVideo && !mainVideo.paused ? mainVideo.currentTime : 0;
      const elapsedSec = state.videoStartTime ? (Date.now() - state.videoStartTime) / 1000 : 0;
      state.interaction.timeSinceAction = Math.round((Date.now() - state.interaction._lastActionTime) / 100) / 10;

      const dataPoint = {
        timestamp: Date.now(),
        videoTime,
        elapsedSec: Math.round(elapsedSec * 10) / 10,
        posture: { ...state.scores },
        gaze: {
          cameraRotation: { ...state.gaze.cameraRotation },
          headDirection: { ...state.gaze.headDirection },
          lookingAt: state.gaze.lookingAt,
          gazeStability: state.gaze.gazeStability,
        },
        engagement: { ...state.engagement },
        emotion: { ...state.emotion },
        audio: { ...state.audio },
        interaction: { lastAction: state.interaction.lastAction, timeSinceAction: state.interaction.timeSinceAction },
        context: { currentText: state.context.currentText },
        hands: { ...state.hands },
      };
      
      state.dataBuffer.push(dataPoint);

      // --- Annotation Mode: Capture Feature Vector ---
      if (state.annotationLabel !== null) {
        const rawFeatures = [...buildFeatureVector(), state.annotationLabel];
        state.annotationBuffer.push(rawFeatures);
      }

      // --- Drowsiness: buffer sample for the 60s sliding window ---
      drowsySample(Date.now());
    }

    // Periodic send (every 5s)
    if (Date.now() - state.lastSendTime > state.sendInterval) {
      if (state.dataBuffer.length > 0) {
        sendPoseData();
      }
      state.lastSendTime = Date.now();
    }
  } catch (e) {
    console.warn("Pose detection frame error:", e.message);
  }
}

// ========== Skeleton Drawing ==========
const CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
];

function drawSkeleton(landmarks) {
  const w = poseCanvas.width;
  const h = poseCanvas.height;

  // Draw connections
  poseCtx.strokeStyle = "rgba(99, 102, 241, 0.7)";
  poseCtx.lineWidth = 2;
  for (const [i, j] of CONNECTIONS) {
    if (landmarks[i] && landmarks[j]) {
      poseCtx.beginPath();
      poseCtx.moveTo(landmarks[i].x * w, landmarks[i].y * h);
      poseCtx.lineTo(landmarks[j].x * w, landmarks[j].y * h);
      poseCtx.stroke();
    }
  }

  // Draw points
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (lm.visibility > 0.5) {
      poseCtx.beginPath();
      poseCtx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
      poseCtx.fillStyle = i <= 10 ? "#f87171" : "#6366f1";
      poseCtx.fill();
    }
  }
}

// ========== Score Calculation ==========
function calculateScores(lm) {
  // Landmark indices: 11=left shoulder, 12=right shoulder, 0=nose, 23=left hip, 24=right hip
  const ls = lm[11], rs = lm[12], nose = lm[0], lh = lm[23], rh = lm[24];

  if (!ls || !rs || !nose || !lh || !rh) return;

  // 1. Shoulder tilt: difference in Y between shoulders (smaller = better)
  const shoulderDiff = Math.abs(ls.y - rs.y);
  const shoulderScore = Math.max(0, Math.min(100, Math.round(100 - shoulderDiff * 500)));

  // 2. Head forward: Z difference between nose and shoulder midpoint
  const shoulderMidZ = (ls.z + rs.z) / 2;
  const headForwardDist = Math.abs(nose.z - shoulderMidZ);
  const headScore = Math.max(0, Math.min(100, Math.round(100 - headForwardDist * 300)));

  // 3. Body lean: angle of spine from hip midpoint to shoulder midpoint
  const hipMidX = (lh.x + rh.x) / 2;
  const hipMidY = (lh.y + rh.y) / 2;
  const shoulderMidX = (ls.x + rs.x) / 2;
  const shoulderMidY = (ls.y + rs.y) / 2;
  const dx = shoulderMidX - hipMidX;
  const dy = shoulderMidY - hipMidY;
  const angle = Math.abs(Math.atan2(dx, -dy) * (180 / Math.PI));
  const leanScore = Math.max(0, Math.min(100, Math.round(100 - angle * 3)));

  // Overall
  const overall = Math.round(shoulderScore * 0.3 + headScore * 0.35 + leanScore * 0.35);

  state.scores = {
    shoulderTilt: shoulderScore,
    headForward: headScore,
    bodyLean: leanScore,
    overall,
  };
}

// ========== UI Update ==========
function getScoreClass(val) {
  if (val >= 70) return "good";
  if (val >= 40) return "warn";
  return "bad";
}

function updateScoreUI() {
  const s = state.scores;

  // Overall circle
  overallScoreNum.textContent = s.overall;
  const offset = 283 - (283 * s.overall) / 100;
  overallProgress.style.strokeDashoffset = offset;
  overallProgress.setAttribute("class", "progress stroke-" + getScoreClass(s.overall));
  overallScoreNum.className = "score-number score-" + getScoreClass(s.overall);

  // Shoulder
  shoulderVal.textContent = s.shoulderTilt;
  shoulderVal.className = "score-val score-" + getScoreClass(s.shoulderTilt);
  shoulderBar.style.width = s.shoulderTilt + "%";
  shoulderBar.className = "score-bar-fill bar-" + getScoreClass(s.shoulderTilt);

  // Head
  headVal.textContent = s.headForward;
  headVal.className = "score-val score-" + getScoreClass(s.headForward);
  headBar.style.width = s.headForward + "%";
  headBar.className = "score-bar-fill bar-" + getScoreClass(s.headForward);

  // Lean
  leanVal.textContent = s.bodyLean;
  leanVal.className = "score-val score-" + getScoreClass(s.bodyLean);
  leanBar.style.width = s.bodyLean + "%";
  leanBar.className = "score-bar-fill bar-" + getScoreClass(s.bodyLean);

  // Status
  statusDataPoints.textContent = state.dataBuffer.length + " pts";

  // Gaze status
  const gazeLabels = { screen: "スクリーン", scoreBoard: "スコアボード", other: "その他" };
  if (statusGaze) statusGaze.textContent = gazeLabels[state.gaze.lookingAt] || "---";
}

// ========== Head Direction from MediaPipe Landmarks ==========
function calculateHeadDirection(lm) {
  // Key landmarks: 0=nose, 2=left eye inner, 5=right eye inner,
  // 7=left ear, 8=right ear, 9=mouth left, 10=mouth right
  const nose = lm[0], leye = lm[2], reye = lm[5];
  const lear = lm[7], rear = lm[8];

  if (!nose || !leye || !reye || !lear || !rear) return;

  // Yaw (left-right rotation): ear-to-ear ratio
  const earDist = Math.sqrt(Math.pow(lear.x - rear.x, 2) + Math.pow(lear.z - rear.z, 2));
  const noseToEarMidX = nose.x - (lear.x + rear.x) / 2;
  const yaw = Math.round(Math.atan2(noseToEarMidX, earDist) * (180 / Math.PI) * 10) / 10;

  // Pitch (up-down): nose Y relative to eye midpoint Y
  const eyeMidY = (leye.y + reye.y) / 2;
  const noseRelY = nose.y - eyeMidY;
  const eyeDist = Math.sqrt(Math.pow(leye.x - reye.x, 2) + Math.pow(leye.y - reye.y, 2));
  const pitch = Math.round(Math.atan2(noseRelY, eyeDist) * (180 / Math.PI) * 10) / 10;

  // Roll (head tilt): angle between eyes
  const roll = Math.round(Math.atan2(reye.y - leye.y, reye.x - leye.x) * (180 / Math.PI) * 10) / 10;

  state.gaze.headDirection = { yaw, pitch, roll };
}

// ========== A-Frame Camera Gaze Detection ==========
function updateGazeFromCamera() {
  // Get camera rotation
  const camera = document.getElementById("vrCamera");
  if (camera) {
    const rot = camera.getAttribute("rotation") || { x: 0, y: 0, z: 0 };
    state.gaze.cameraRotation = {
      x: Math.round(rot.x * 10) / 10,
      y: Math.round(rot.y * 10) / 10,
      z: Math.round(rot.z * 10) / 10,
    };

    // Detect movement for engagement
    const dx = Math.abs(rot.x - state._prevCamRot.x);
    const dy = Math.abs(rot.y - state._prevCamRot.y);
    if (dx > 0.5 || dy > 0.5) {
      state._lastMovementTime = Date.now();
    }
    state._prevCamRot = { x: rot.x, y: rot.y };
  }

  // Get raycaster intersection
  const cursor = document.getElementById("gazeCursor");
  if (cursor && cursor.components.raycaster) {
    const intersections = cursor.components.raycaster.intersections;
    if (intersections && intersections.length > 0) {
      const hit = intersections[0].object.el;
      const label = hit.getAttribute("data-label");
      state.gaze.lookingAt = label || "other";
    } else {
      state.gaze.lookingAt = "other";
    }
  }

  // Calculate gaze stability (how steady gaze target has been)
  state._gazeHistory.push(state.gaze.lookingAt);
  if (state._gazeHistory.length > 30) state._gazeHistory.shift(); // keep ~30 frames
  const modeTarget = state._gazeHistory.reduce((acc, v) => {
    acc[v] = (acc[v] || 0) + 1;
    return acc;
  }, {});
  const maxCount = Math.max(...Object.values(modeTarget));
  state.gaze.gazeStability = Math.round((maxCount / state._gazeHistory.length) * 100) / 100;
}

// ========== Engagement Estimation ==========
function calculateEngagement() {
  const s = state.scores;
  const g = state.gaze;

  state.engagement.isLookingAtScreen = g.lookingAt === "screen";
  state.engagement.postureQuality = s.overall >= 70 ? "good" : s.overall >= 40 ? "warn" : "bad";
  state.engagement.secondsSinceLastMovement =
    Math.round((Date.now() - state._lastMovementTime) / 100) / 10;
}

// ========== Data Sending ==========
async function sendPoseData() {
  if (state.dataBuffer.length === 0) return;
  const count = state.dataBuffer.length;
  const payload = {
    sessionId: state.sessionId,
    dataPoints: [...state.dataBuffer],
  };
  state.dataBuffer = [];
  try {
    const resp = await fetch("/api/pose-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    console.log(`[Pose] Sent ${count} points → total: ${data.totalPoints}`);
    showToast(`💾 ${count}件のデータを保存 (計${data.totalPoints}件)`, 2000);
  } catch (e) {
    console.error("Pose data send error:", e);
    // Re-add failed data to buffer
    state.dataBuffer.unshift(...payload.dataPoints);
    showToast("⚠️ データ送信失敗。再試行します。", 2000);
  }
}

// ========== AI Supporter Logic ==========
function speakMessage(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ja-JP";
  utterance.volume = 0.5;
  utterance.rate = 1.1;
  window.speechSynthesis.speak(utterance);
}

function setDefaultAIMessage() {
  aiSupporterPanel.classList.remove("hidden", "active-warning");
  aiMessageText.textContent = "あなたの学習をサポートします。";
  aiActionsContainer.innerHTML = "";
  state.aiTriggerTracker.currentPattern = null;
}

function showAIMessage(pattern, text, actions = []) {
  // 8s cooldown between AI messages to avoid spamming
  if (Date.now() - state.aiTriggerTracker.lastMessageTime < 8000) return;
  
  aiSupporterPanel.classList.remove("hidden");
  aiMessageText.textContent = text;
  aiActionsContainer.innerHTML = "";
  
  if (pattern === 'B' || pattern === 'C' || pattern === 'E') {
    aiSupporterPanel.classList.add("active-warning");
  } else {
    aiSupporterPanel.classList.remove("active-warning");
  }
  
  actions.forEach(action => {
    const btn = document.createElement("button");
    btn.className = "btn " + (action.type || "btn-primary");
    btn.textContent = action.label;
    btn.onclick = action.onClick;
    aiActionsContainer.appendChild(btn);
  });
  
  // Close button always present
  const closeBtn = document.createElement("button");
  closeBtn.className = "btn btn-danger";
  closeBtn.textContent = "閉じる";
  closeBtn.onclick = setDefaultAIMessage;
  aiActionsContainer.appendChild(closeBtn);
  
  speakMessage(text);
  state.aiTriggerTracker.lastMessageTime = Date.now();
  state.aiTriggerTracker.currentPattern = pattern;
  state.mlPredHistory = []; // reset votes so a stale majority can't re-trigger
  
  if (pattern === 'D') {
    setTimeout(setDefaultAIMessage, 5000);
  }
}

function checkAITriggers() {
  if (!state.webcamRunning || !state.mlModel || !state.mlScaler) return;

  const now = Date.now();

  // Throttle inference: detectPose runs at ~30fps but predicting every frame
  // is wasteful and makes the trigger timers hypersensitive.
  if (now - state._lastPredictTime < ML_PREDICT_INTERVAL) return;
  state._lastPredictTime = now;

  const tracker = state.aiTriggerTracker;

  const rawFeatures = buildFeatureVector();

  // Scale features with z-clipping.
  // Clipping to [-ML_Z_CLIP, +ML_Z_CLIP] prevents features that fall outside
  // the (narrow) training distribution from exploding into huge z-scores and
  // dominating the prediction — this was the root cause of the model always
  // predicting "Confusion" at runtime.
  const scaledFeatures = rawFeatures.map((val, i) => {
    const mean = state.mlScaler.mean[i];
    const scale = state.mlScaler.scale[i];
    if (!(scale > 1e-6)) return 0;
    const z = (val - mean) / scale;
    return Math.max(-ML_Z_CLIP, Math.min(ML_Z_CLIP, z));
  });

  // Predict (single sample)
  const { classId, prob } = tf.tidy(() => {
    const inputTensor = tf.tensor2d([scaledFeatures]);
    const probs = state.mlModel.predict(inputTensor).dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    state.lastProbs = Array.from(probs); // for the live monitor panel
    return { classId: best, prob: probs[best] };
  });

  // --- Confidence gate + majority-vote smoothing ---
  // Low-confidence predictions count as Normal (0) instead of triggering.
  const votedClass = prob >= ML_PROB_THRESHOLD ? classId : 0;
  state.mlPredHistory.push(votedClass);
  if (state.mlPredHistory.length > ML_HISTORY_SIZE) state.mlPredHistory.shift();

  // Effective class = the class that got a clear majority in the window.
  let effectiveClass = 0;
  if (state.mlPredHistory.length >= ML_HISTORY_SIZE) {
    const counts = {};
    for (const c of state.mlPredHistory) counts[c] = (counts[c] || 0) + 1;
    for (const [c, n] of Object.entries(counts)) {
      if (n >= ML_MAJORITY_MIN && parseInt(c) !== 0) effectiveClass = parseInt(c);
    }
  }

  // Prediction runs whenever tracking is on (so the monitor always moves).
  // INTERVENTIONS (pause / rewind suggestions) only fire while the video
  // is actually playing.
  const videoActive = state.videoLoaded && !mainVideo.paused;
  if (!videoActive) {
    tracker.confusionStart = null;
    tracker.distractionStart = null;
    tracker.badPostureStart = null;
    tracker.flowStart = null;
    return;
  }

  applyTriggerLogic(effectiveClass, now, tracker);
}

function applyTriggerLogic(classId, now, tracker) {
  {
    // Labels: 0: Normal, 1: Confusion, 2: Distraction, 3: BadPosture, 4: Flow
    // We apply cooldown / duration logic manually to avoid spamming
    if (classId === 1) { // Confusion
      if (!tracker.confusionStart) tracker.confusionStart = now;
      else if (now - tracker.confusionStart > 3000) {
        flashStateBar(1);
        showAIMessage('A', "少し難しいですか？もう一度見直すのもおすすめです。", [
          { label: "⏪ 10秒戻る", onClick: () => { mainVideo.currentTime = Math.max(0, mainVideo.currentTime - 10); setDefaultAIMessage(); tracker.confusionStart = null; } }
        ]);
        tracker.confusionStart = null;
      }
    } else {
      tracker.confusionStart = null;
    }
    
    if (classId === 2) { // Distraction
      if (!tracker.distractionStart) tracker.distractionStart = now;
      else if (now - tracker.distractionStart > 5000) {
        flashStateBar(2);
        mainVideo.pause();
        showAIMessage('B', "少し疲れが見えますね。リフレッシュしましょう！", [
          { label: "✅ 了解", onClick: () => { setDefaultAIMessage(); tracker.distractionStart = null; mainVideo.play(); } }
        ]);
        tracker.distractionStart = null;
      }
    } else {
      tracker.distractionStart = null;
    }
    
    if (classId === 3) { // Bad Posture
      if (!tracker.badPostureStart) tracker.badPostureStart = now;
      else if (now - tracker.badPostureStart > 5000) {
        flashStateBar(3);
        showAIMessage('C', "姿勢が悪くなっています。少し背筋を伸ばしてみましょう。");
        tracker.badPostureStart = null;
      }
    } else {
      tracker.badPostureStart = null;
    }
    
    if (classId === 4) { // Flow
      if (!tracker.flowStart) tracker.flowStart = now;
      else if (now - tracker.flowStart > 15000) {
        flashStateBar(4);
        showAIMessage('D', "素晴らしい集中力ですね！");
        tracker.flowStart = null;
      }
    } else {
      tracker.flowStart = null;
    }
  }
}

// ========== Video & VTT Loading ==========
function parseVTT(vttText) {
  const lines = vttText.split(/\r?\n/);
  const subtitles = [];
  let currentSub = null;
  const timeRegex = /(\d{2}:)?(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d{2}:)?(\d{2}):(\d{2})\.(\d{3})/;

  const parseTime = (h, m, s, ms) => {
    return (parseInt(h || 0) * 3600) + (parseInt(m) * 60) + parseInt(s) + (parseInt(ms) / 1000);
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === 'WEBVTT' || line === '') continue;
    
    const timeMatch = line.match(timeRegex);
    if (timeMatch) {
      if (currentSub) subtitles.push(currentSub);
      currentSub = {
        start: parseTime(timeMatch[1], timeMatch[2], timeMatch[3], timeMatch[4]),
        end: parseTime(timeMatch[5], timeMatch[6], timeMatch[7], timeMatch[8]),
        text: ''
      };
    } else if (currentSub) {
      currentSub.text += (currentSub.text ? ' ' : '') + line;
    }
  }
  if (currentSub) subtitles.push(currentSub);
  return subtitles;
}

vttInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.context._subtitles = parseVTT(e.target.result);
    showToast("📝 字幕ファイル(VTT)を読み込みました");
  };
  reader.readAsText(file);
});

mainVideo.addEventListener("timeupdate", () => {
  const t = mainVideo.currentTime;
  const sub = state.context._subtitles.find(s => t >= s.start && t <= s.end);
  state.context.currentText = sub ? sub.text : "";
});

function loadVideo() {
  const file = videoInput.files[0];
  if (!file) {
    showToast("⚠️ 動画ファイルを選択してください");
    return;
  }
  const url = URL.createObjectURL(file);
  mainVideo.src = url;
  mainVideo.play();
  state.videoLoaded = true;
  state.videoStartTime = Date.now();
  state.interaction.lastAction = "play";
  state.interaction._lastActionTime = Date.now();
  showToast("🎬 動画を再生開始しました");
}

// ========== Event Listeners ==========
loadBtn.addEventListener("click", loadVideo);
pauseBtn.addEventListener("click", () => {
  if (!state.videoLoaded) return;
  if (mainVideo.paused) {
    mainVideo.play();
  } else {
    mainVideo.pause();
  }
});
startPoseBtn.addEventListener("click", startWebcam);
stopPoseBtn.addEventListener("click", stopWebcam);

// Video Interaction listeners
mainVideo.addEventListener("play", () => {
  state.interaction.lastAction = "play";
  state.interaction._lastActionTime = Date.now();
  if (pauseBtn) pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> 一時停止';
});
mainVideo.addEventListener("pause", () => {
  state.interaction.lastAction = "pause";
  state.interaction._lastActionTime = Date.now();
  if (pauseBtn) pauseBtn.innerHTML = '<span class="btn-icon">▶</span> 再開';
});
mainVideo.addEventListener("seeked", () => {
  state.interaction.lastAction = "seeked";
  state.interaction._lastActionTime = Date.now();
});

// Status bar session
statusSession.textContent = state.sessionId.slice(0, 8);

async function loadML() {
  try {
    showToast("🤖 MLモデルを読み込み中...");
    const modelUrl = "/public/model/model.json";
    state.mlModel = await tf.loadLayersModel(modelUrl);
    
    const scalerResp = await fetch("/public/model/scaler_params.json");
    state.mlScaler = await scalerResp.json();
    showToast("✅ MLモデル準備完了");
  } catch (e) {
    console.error("ML Load error:", e);
    showToast("⚠️ MLモデルの読み込みに失敗しました");
  }
}

// ========== Annotation Mode ==========
function initAnnotationMode() {
  const labelMap = {
    '0': 'Normal',
    '1': 'Confusion',
    '2': 'Distraction',
    '3': 'Bad Posture',
    '4': 'Flow'
  };

  const indicator = document.createElement('div');
  indicator.id = 'annotation-indicator';
  indicator.style.position = 'fixed';
  indicator.style.top = '20px';
  indicator.style.left = '50%';
  indicator.style.transform = 'translateX(-50%)';
  indicator.style.padding = '10px 20px';
  indicator.style.backgroundColor = 'rgba(239, 68, 68, 0.9)';
  indicator.style.color = '#fff';
  indicator.style.borderRadius = '8px';
  indicator.style.fontWeight = 'bold';
  indicator.style.fontSize = '18px';
  indicator.style.zIndex = '9999';
  indicator.style.display = 'none';
  indicator.style.pointerEvents = 'none';
  document.body.appendChild(indicator);

  window.addEventListener('keydown', (e) => {
    if (['0', '1', '2', '3', '4'].includes(e.key)) {
      state.annotationLabel = parseInt(e.key);
      indicator.textContent = `🔴 記録中: ${e.key} (${labelMap[e.key]})`;
      indicator.style.display = 'block';
    }
  });

  window.addEventListener('keyup', (e) => {
    if (['0', '1', '2', '3', '4'].includes(e.key)) {
      state.annotationLabel = null;
      indicator.style.display = 'none';
    }
  });

  // Send annotation data periodically
  setInterval(async () => {
    if (state.annotationBuffer.length === 0) return;
    const rows = [...state.annotationBuffer];
    state.annotationBuffer = [];
    
    try {
      const resp = await fetch('/api/annotation-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows })
      });
      if (resp.ok) {
        console.log(`[Annotation] Sent ${rows.length} records`);
      }
    } catch (e) {
      console.error('Annotation send error:', e);
      state.annotationBuffer.unshift(...rows);
    }
  }, 2000); // Check every 2s
}

initAnnotationMode();

// ==================================================================
// ========== Drowsiness Detection (time-window features) ==========
// ==================================================================
// Drowsiness is a TEMPORAL pattern, not a single-frame state. We keep a
// 60-second sliding window of cheap per-frame signals and derive research-
// backed features from it: PERCLOS, blink rate/duration, yawns, head nods.
// A personal baseline for eye openness is calibrated during the first 45s.

const DROWSY = {
  WINDOW_MS: 60000,        // sliding window length
  BASELINE_MS: 45000,      // calibration period for personal eye baseline
  FEATURE_INTERVAL: 1000,  // recompute window features every 1s
  ROW_INTERVAL: 5000,      // save one labelled row every 5s while recording
  PREDICT_INTERVAL: 2000,  // run drowsiness model every 2s
  SMOOTH_N: 15,            // average predictions over ~30s
  SCORE_THRESHOLD: 0.6,    // smoothed drowsiness score that triggers a break
  BREAK_COOLDOWN: 120000,  // min 2 min between break suggestions
  FEATURE_NAMES: [
    "perclos", "blinkRate", "meanBlinkDur", "longBlinks", "yawns",
    "nods", "pitchStd", "pitchDrift", "postureStd", "elapsedMin",
  ],
};

const drowsy = {
  samples: [],        // {t, eye, jaw, pitch, overall} @5fps, last 60s
  blinks: [],         // {t, dur(s)} within window
  yawns: [],          // timestamps within window
  nods: [],           // timestamps within window
  baselineEye: null,
  _baselineBuf: [],
  _inBlink: false, _blinkStart: 0,
  _inYawn: false, _yawnStart: 0,
  _lastNod: 0,
  features: null,     // latest 10-dim feature vector
  label: 0,           // 0=覚醒, 1=少し眠い, 2=眠い
  recording: false,
  rowsSent: 0,
  model: null, scaler: null,
  predSmoothed: null, // {probs:[3], score}
  _predHist: [],
  trackStart: null,
  _lastFeat: 0, _lastRow: 0, _lastPred: 0, _lastBreak: 0, _lastReminder: 0,
};

function _mean(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
function _std(a) {
  if (a.length < 2) return 0;
  const m = _mean(a);
  return Math.sqrt(_mean(a.map(v => (v - m) * (v - m))));
}
function _median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

function resetDrowsyTracking() {
  drowsy.samples = []; drowsy.blinks = []; drowsy.yawns = []; drowsy.nods = [];
  drowsy.baselineEye = null; drowsy._baselineBuf = [];
  drowsy._inBlink = false; drowsy._inYawn = false;
  drowsy.features = null; drowsy._predHist = []; drowsy.predSmoothed = null;
  drowsy.trackStart = Date.now();
}

// Full-frame-rate display buffers (viz only; ML features still use 5fps samples)
const VIZ_STREAM_MS = 60000;
const vizStream = { eye: [], pitch: [] };

/** Per-frame (~30fps): blink & yawn state machines need full frame rate. */
function drowsyRealtimeUpdate(now) {
  if (!drowsy.trackStart) drowsy.trackStart = now;
  const eye = state.emotion.eyeOpenness ?? 1;
  const jaw = state.emotion.jawOpen ?? 0;

  // Feed the smooth-display buffers at camera frame rate
  vizStream.eye.push({ t: now, v: eye });
  vizStream.pitch.push({ t: now, v: state.gaze.headDirection?.pitch ?? 0 });
  const vcut = now - VIZ_STREAM_MS;
  while (vizStream.eye.length && vizStream.eye[0].t < vcut) vizStream.eye.shift();
  while (vizStream.pitch.length && vizStream.pitch[0].t < vcut) vizStream.pitch.shift();

  // Personal baseline (first 45s of tracking)
  if (drowsy.baselineEye === null) {
    drowsy._baselineBuf.push(eye);
    if (now - drowsy.trackStart >= DROWSY.BASELINE_MS) {
      drowsy.baselineEye = _mean(drowsy._baselineBuf);
      drowsy._baselineBuf = [];
      showToast(`👁 個人ベースライン確定 (目の開き平常値: ${drowsy.baselineEye.toFixed(2)})`);
    }
  }
  const base = drowsy.baselineEye ?? 0.9;

  // Blink state machine (relative threshold: 55% of personal baseline)
  const blinkThr = base * 0.55;
  if (!drowsy._inBlink && eye < blinkThr) {
    drowsy._inBlink = true; drowsy._blinkStart = now;
  } else if (drowsy._inBlink && eye >= blinkThr) {
    drowsy._inBlink = false;
    const dur = (now - drowsy._blinkStart) / 1000;
    if (dur >= 0.05 && dur < 3) drowsy.blinks.push({ t: now, dur });
  }

  // Yawn state machine (mouth wide open, sustained > 1.2s)
  if (!drowsy._inYawn && jaw > 0.5) {
    drowsy._inYawn = true; drowsy._yawnStart = now;
  } else if (drowsy._inYawn && jaw < 0.3) {
    drowsy._inYawn = false;
    if (now - drowsy._yawnStart > 1200) drowsy.yawns.push(now);
  }
}

/** ~5fps: buffer window samples, detect nods, refresh features. */
function drowsySample(now) {
  drowsy.samples.push({
    t: now,
    eye: state.emotion.eyeOpenness ?? 1,
    jaw: state.emotion.jawOpen ?? 0,
    pitch: state.gaze.headDirection?.pitch ?? 0,
    overall: state.scores.overall || 0,
  });
  const cutoff = now - DROWSY.WINDOW_MS;
  while (drowsy.samples.length && drowsy.samples[0].t < cutoff) drowsy.samples.shift();
  drowsy.blinks = drowsy.blinks.filter(b => b.t >= cutoff);
  drowsy.yawns = drowsy.yawns.filter(t => t >= cutoff);
  drowsy.nods = drowsy.nods.filter(t => t >= cutoff);

  // Head-nod detection: pitch suddenly drops (chin down = pitch up) vs the
  // median of the previous ~2s
  const pitch = state.gaze.headDirection?.pitch ?? 0;
  const recent = drowsy.samples
    .filter(s => s.t > now - 2500 && s.t < now - 300)
    .map(s => s.pitch);
  if (recent.length >= 5 && pitch - _median(recent) > 12 && now - drowsy._lastNod > 4000) {
    drowsy._lastNod = now;
    drowsy.nods.push(now);
  }

  if (now - drowsy._lastFeat >= DROWSY.FEATURE_INTERVAL) {
    drowsy._lastFeat = now;
    computeDrowsyFeatures(now);
    maybeSendDrowsyRow(now);
    maybePredictDrowsy(now);
  }
}

function computeDrowsyFeatures(now) {
  const S = drowsy.samples;
  if (S.length < 25) return; // need ~5s of samples first
  const base = drowsy.baselineEye ?? 0.9;
  const closedThr = base * 0.6;

  const winSec = Math.max((S[S.length - 1].t - S[0].t) / 1000, 1);
  const perclos = S.filter(s => s.eye < closedThr).length / S.length;
  const blinkRate = drowsy.blinks.length * (60 / winSec);
  const meanBlinkDur = drowsy.blinks.length ? _mean(drowsy.blinks.map(b => b.dur)) : 0;
  const longBlinks = drowsy.blinks.filter(b => b.dur > 0.4).length;

  const pitches = S.map(s => s.pitch);
  const pitchStd = _std(pitches);
  const mid = S[0].t + (winSec * 1000) / 2;
  const early = S.filter(s => s.t < mid).map(s => s.pitch);
  const late = S.filter(s => s.t >= mid).map(s => s.pitch);
  const pitchDrift = early.length && late.length ? _mean(late) - _mean(early) : 0;

  const postureStd = _std(S.map(s => s.overall));
  const elapsedMin = (now - drowsy.trackStart) / 60000;

  drowsy.features = [
    perclos, blinkRate, meanBlinkDur, longBlinks, drowsy.yawns.length,
    drowsy.nods.length, pitchStd, pitchDrift, postureStd, elapsedMin,
  ];
}

function maybeSendDrowsyRow(now) {
  if (!drowsy.recording || !drowsy.features) return;
  if (now - drowsy._lastRow < DROWSY.ROW_INTERVAL) return;
  drowsy._lastRow = now;

  const row = [...drowsy.features.map(v => Math.round(v * 10000) / 10000), drowsy.label];
  fetch('/api/drowsiness-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows: [row] }),
  }).then(r => { if (r.ok) drowsy.rowsSent++; }).catch(() => {});

  // Every 5 min: remind the user to update their self-reported label
  if (now - drowsy._lastReminder > 300000) {
    drowsy._lastReminder = now;
    showToast('😴 眠気レベルは今も正しいですか?パネルで更新してください', 5000);
  }
}

async function loadDrowsyModel() {
  try {
    drowsy.model = await tf.loadLayersModel('/public/model_drowsy/model.json');
    const resp = await fetch('/public/model_drowsy/scaler_params.json');
    drowsy.scaler = await resp.json();
    showToast('✅ 眠気予測モデル準備完了');
  } catch (e) {
    // Model not trained yet — the monitor panel will show 未学習
    drowsy.model = null; drowsy.scaler = null;
    console.log('[Drowsy] model not available yet (train with ml/train_drowsy.py)');
  }
}

function maybePredictDrowsy(now) {
  if (!drowsy.model || !drowsy.scaler || !drowsy.features) return;
  if (now - drowsy._lastPred < DROWSY.PREDICT_INTERVAL) return;
  drowsy._lastPred = now;

  const scaled = drowsy.features.map((val, i) => {
    const mean = drowsy.scaler.mean[i];
    const scale = drowsy.scaler.scale[i];
    if (!(scale > 1e-6)) return 0;
    return Math.max(-3, Math.min(3, (val - mean) / scale));
  });

  const probs = tf.tidy(() => {
    const t = tf.tensor2d([scaled]);
    return Array.from(drowsy.model.predict(t).dataSync());
  });

  drowsy._predHist.push(probs);
  if (drowsy._predHist.length > DROWSY.SMOOTH_N) drowsy._predHist.shift();
  const n = drowsy._predHist.length;
  const avg = [0, 0, 0];
  for (const p of drowsy._predHist) for (let i = 0; i < 3; i++) avg[i] += p[i] / n;
  // Drowsiness score: 少し眠い counts half, 眠い counts full
  const score = avg[1] * 0.5 + (avg[2] || 0);
  drowsy.predSmoothed = { probs: avg, score };

  // Break suggestion: sustained drowsiness while the video is playing
  const videoActive = state.videoLoaded && !mainVideo.paused;
  if (videoActive && n >= DROWSY.SMOOTH_N && score > DROWSY.SCORE_THRESHOLD &&
      now - drowsy._lastBreak > DROWSY.BREAK_COOLDOWN) {
    drowsy._lastBreak = now;
    if (vizEls) flashEl(vizEls.drowsyArea);
    mainVideo.pause();
    showAIMessage('E', '眠気のサインが続いています。動画を止めました。少し休憩しませんか?', [
      { label: '☕ 休憩する', onClick: () => { setDefaultAIMessage(); } },
      { label: '▶ 続ける', onClick: () => { setDefaultAIMessage(); mainVideo.play(); } },
    ]);
  }
}

// ==================================================================
// ========== ML Live Monitor Panel (right side, for demos) =========
// ==================================================================
const VIZ_CSS = `
#ml-viz-panel {
  position: fixed; top: 80px; right: 20px; bottom: 64px; width: 350px;
  z-index: 2000; overflow-y: auto; padding: 16px;
  /* NOTE: no backdrop-filter here — blurring the live WebGL scene behind
     the panel forces a full re-blur every A-Frame frame and tanks FPS */
  background: rgba(13, 17, 30, 0.97);
  border: 1px solid rgba(255,255,255,0.09); border-radius: 16px;
  color: #f1f5f9; font-size: 13px;
  scrollbar-width: thin;
  contain: layout paint; /* isolate repaints from the rest of the page */
}
#ml-viz-panel.hidden { display: none; }
#ml-viz-panel .viz-title {
  font-size: 13px; font-weight: 700; color: #cbd5e1;
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 12px;
}
#ml-viz-panel .viz-section { margin-bottom: 14px; }
#ml-viz-panel .viz-label {
  font-size: 12.5px; color: #a8b5c7; margin-bottom: 5px;
  display: flex; justify-content: space-between; align-items: baseline;
}
#ml-viz-panel .viz-label .viz-stat { color: #e2e8f0; font-weight: 600; }
#ml-viz-panel canvas {
  width: 100%; border-radius: 8px; background: rgba(255,255,255,0.04);
  display: block;
}
#ml-viz-panel .viz-gauge {
  height: 12px; border-radius: 6px; background: rgba(255,255,255,0.06);
  overflow: hidden; margin-bottom: 6px;
}
#ml-viz-panel .viz-gauge > div {
  height: 100%; width: 100%; border-radius: 6px;
  background: linear-gradient(90deg, #34d399, #fbbf24, #f87171);
  /* scaleX runs on the GPU compositor — no layout/repaint per frame */
  transform: scaleX(0); transform-origin: left center; will-change: transform;
}
#ml-viz-panel .viz-chips { display: flex; flex-wrap: wrap; gap: 5px; }
#ml-viz-panel .viz-chip {
  background: rgba(255,255,255,0.07); border-radius: 6px; padding: 3px 8px;
  font-size: 12px; color: #d5dde8;
}
#ml-viz-panel .viz-chip b { color: #fff; }
#ml-viz-panel .viz-bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
#ml-viz-panel .viz-bar-name { width: 64px; font-size: 12px; color: #d5dde8; }
#ml-viz-panel .viz-bar-track {
  flex: 1; height: 10px; border-radius: 5px; background: rgba(255,255,255,0.07); overflow: hidden;
}
#ml-viz-panel .viz-bar-fill {
  height: 100%; width: 100%; background: #6366f1;
  transform: scaleX(0); transform-origin: left center; will-change: transform;
}
#ml-viz-panel .viz-bar-row.viz-top .viz-bar-fill { background: #34d399; }
#ml-viz-panel .viz-bar-val { width: 34px; text-align: right; font-size: 12px; color: #a8b5c7; }
#ml-viz-panel .viz-drowsy-score { font-size: 26px; font-weight: 700; text-align: center; margin: 4px 0; }
#ml-viz-panel .viz-muted { color: #8492a6; font-size: 12px; }
#ml-viz-panel .viz-rec-labels { display: flex; gap: 5px; margin-bottom: 8px; }
#ml-viz-panel .viz-rec-labels button {
  flex: 1; padding: 7px 2px; font-size: 12px; border-radius: 8px; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); color: #d5dde8;
}
#ml-viz-panel .viz-rec-labels button.active {
  background: #6366f1; border-color: #6366f1; color: #fff; font-weight: 700;
}
#ml-viz-panel #vizRecToggle {
  width: 100%; padding: 8px; border-radius: 8px; border: none; cursor: pointer;
  background: rgba(52, 211, 153, 0.15); color: #34d399; font-weight: 700; font-size: 12px;
}
#ml-viz-panel #vizRecToggle.recording {
  background: rgba(248, 113, 113, 0.18); color: #f87171;
  animation: vizPulse 1.2s ease-in-out infinite;
}
@keyframes vizPulse { 0%,100% { opacity: 1; } 50% { opacity: .55; } }
#ml-viz-panel .viz-divider { height: 1px; background: rgba(255,255,255,0.08); margin: 12px 0; }
/* Drowsiness isn't part of this presentation — hidden for a cleaner panel.
   Data collection/training code is untouched; delete this rule to bring it back. */
#ml-viz-panel #vizDrowsyBlock { display: none; }
/* Brief highlight pulse so "why the AI reacted" is visible without narration */
@keyframes vizFlash {
  0% { box-shadow: 0 0 0 0 rgba(251,191,36,0.85); background: rgba(251,191,36,0.35); }
  100% { box-shadow: 0 0 0 10px rgba(251,191,36,0); background: rgba(255,255,255,0.06); }
}
#ml-viz-panel .viz-flash { animation: vizFlash 900ms ease-out; }
#ml-viz-panel .viz-bar-row.viz-flash .viz-bar-track { animation: vizFlash 900ms ease-out; }
/* On the ML page, move the posture panel to the left so the monitor owns the right side */
#posture-panel { right: auto; left: 20px; top: 360px; }
@media (max-width: 900px) { #ml-viz-panel { display: none; } }
`;

const STATE_CLASS_INFO = [
  { name: '通常', color: '#94a3b8' },
  { name: '困惑', color: '#fbbf24' },
  { name: '脇見', color: '#f87171' },
  { name: '姿勢', color: '#fb923c' },
  { name: '集中', color: '#34d399' },
];

let vizEls = null;

function initVizPanel() {
  const style = document.createElement('style');
  style.textContent = VIZ_CSS;
  document.head.appendChild(style);

  const panel = document.createElement('div');
  panel.id = 'ml-viz-panel';
  panel.className = 'hidden';
  panel.innerHTML = `
    <div class="viz-title">📈 ML Live Monitor</div>

    <div class="viz-section">
      <div class="viz-label"><span>👁 目の開き (直近60秒)</span><span class="viz-stat" id="vizEyeStat">--</span></div>
      <canvas id="vizEyeCanvas" width="288" height="56"></canvas>
    </div>

    <div class="viz-section">
      <div class="viz-label"><span>↕️ 頭のピッチ / ノッド</span><span class="viz-stat" id="vizPitchStat">--</span></div>
      <canvas id="vizPitchCanvas" width="288" height="48"></canvas>
    </div>

    <div class="viz-section">
      <div class="viz-label"><span>😪 PERCLOS (閉眼時間率)</span><span class="viz-stat" id="vizPerclosVal">--</span></div>
      <div class="viz-gauge"><div id="vizPerclosBar"></div></div>
      <div class="viz-chips" id="vizChips">
        <span class="viz-chip">まばたき <b id="vizChipBlink">--</b>/分</span>
        <span class="viz-chip">平均 <b id="vizChipDur">--</b>ms</span>
        <span class="viz-chip">ロング <b id="vizChipLong">--</b></span>
        <span class="viz-chip">あくび <b id="vizChipYawn">--</b></span>
        <span class="viz-chip">ノッド <b id="vizChipNod">--</b></span>
        <span class="viz-chip">ドリフト <b id="vizChipDrift">--</b>°</span>
        <span class="viz-chip">経過 <b id="vizChipElapsed">--</b>分</span>
      </div>
    </div>

    <div class="viz-divider"></div>

    <div class="viz-section">
      <div class="viz-label"><span>🧠 状態モデル (リアルタイム判定)</span></div>
      <div id="vizStateBars"></div>
    </div>

    <div id="vizDrowsyBlock">
      <div class="viz-section">
        <div class="viz-label"><span>😴 眠気モデル</span></div>
        <div id="vizDrowsyNone" class="viz-muted">モデル未学習 (train_drowsy.py で学習)</div>
        <div id="vizDrowsyArea" style="display:none">
          <div class="viz-drowsy-score"><span id="vizDrowsyScore">--</span><span style="font-size:12px;color:#94a3b8"> /100 <span id="vizDrowsyLabel"></span></span></div>
          <div class="viz-gauge"><div id="vizDrowsyBar"></div></div>
          <div class="viz-muted" style="margin-top:4px">覚醒 <span id="vizDrowsyP0">--</span>% / 少し眠い <span id="vizDrowsyP1">--</span>% / 眠い <span id="vizDrowsyP2">--</span>%</div>
        </div>
      </div>

      <div class="viz-divider"></div>

      <div class="viz-section">
        <div class="viz-label"><span>🎙 眠気データ記録 (自己申告)</span><span class="viz-stat" id="vizRowCount"></span></div>
        <div class="viz-rec-labels">
          <button data-dlabel="0" class="active">😀 覚醒</button>
          <button data-dlabel="1">😑 少し眠い</button>
          <button data-dlabel="2">😴 眠い</button>
        </div>
        <button id="vizRecToggle">● 記録開始</button>
      </div>
    </div>
  `;
  document.body.appendChild(panel);

  // State-model probability bars
  const barsHost = panel.querySelector('#vizStateBars');
  for (const info of STATE_CLASS_INFO) {
    const row = document.createElement('div');
    row.className = 'viz-bar-row';
    row.innerHTML = `
      <span class="viz-bar-name">${info.name}</span>
      <div class="viz-bar-track"><div class="viz-bar-fill" style="background:${info.color}"></div></div>
      <span class="viz-bar-val">--</span>`;
    barsHost.appendChild(row);
  }

  // Label buttons
  panel.querySelectorAll('[data-dlabel]').forEach(btn => {
    btn.addEventListener('click', () => {
      drowsy.label = parseInt(btn.dataset.dlabel);
      panel.querySelectorAll('[data-dlabel]').forEach(b => b.classList.toggle('active', b === btn));
    });
  });

  // Record toggle
  const recBtn = panel.querySelector('#vizRecToggle');
  recBtn.addEventListener('click', () => {
    drowsy.recording = !drowsy.recording;
    recBtn.classList.toggle('recording', drowsy.recording);
    recBtn.textContent = drowsy.recording ? '⏺ 記録中... (クリックで停止)' : '● 記録開始';
    showToast(drowsy.recording
      ? '😴 眠気データの記録を開始しました (5秒ごとに1行保存)'
      : `⏹ 記録停止 (合計 ${drowsy.rowsSent} 行保存)`);
  });

  const q = (id) => panel.querySelector('#' + id);
  vizEls = {
    panel,
    eyeStat: q('vizEyeStat'),
    eyeCanvas: q('vizEyeCanvas'),
    pitchStat: q('vizPitchStat'),
    pitchCanvas: q('vizPitchCanvas'),
    perclosVal: q('vizPerclosVal'),
    perclosBar: q('vizPerclosBar'),
    chipBlink: q('vizChipBlink'), chipDur: q('vizChipDur'),
    chipLong: q('vizChipLong'), chipYawn: q('vizChipYawn'),
    chipNod: q('vizChipNod'), chipDrift: q('vizChipDrift'),
    chipElapsed: q('vizChipElapsed'),
    stateBars: panel.querySelectorAll('#vizStateBars .viz-bar-row'),
    stateFills: [...panel.querySelectorAll('#vizStateBars .viz-bar-fill')],
    stateVals: [...panel.querySelectorAll('#vizStateBars .viz-bar-val')],
    drowsyNone: q('vizDrowsyNone'), drowsyArea: q('vizDrowsyArea'),
    drowsyScore: q('vizDrowsyScore'), drowsyLabel: q('vizDrowsyLabel'),
    drowsyBar: q('vizDrowsyBar'),
    drowsyP0: q('vizDrowsyP0'), drowsyP1: q('vizDrowsyP1'), drowsyP2: q('vizDrowsyP2'),
    rowCount: q('vizRowCount'),
  };
}

/** Write textContent only when the value actually changed (avoids DOM churn). */
function setText(el, text) {
  if (el && el.textContent !== text) el.textContent = text;
}

/** Retrigger the vizFlash pulse on an element (restarts the CSS animation). */
function flashEl(el) {
  if (!el) return;
  el.classList.remove('viz-flash');
  void el.offsetWidth; // force reflow so the animation restarts
  el.classList.add('viz-flash');
}

/** Highlight the state-model bar row that just caused an AI message to fire. */
function flashStateBar(classId) {
  if (!vizEls || !vizEls.stateBars[classId]) return;
  flashEl(vizEls.stateBars[classId]);
}

function drawSparkline(canvas, points, opts = {}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (points.length < 2) {
    ctx.fillStyle = '#475569';
    ctx.font = '11px sans-serif';
    ctx.fillText('データ収集中...', 10, h / 2 + 4);
    return;
  }
  // Adaptive downsample: stride the old part, keep the newest ~100 points at
  // full resolution so the leading edge of the line stays perfectly smooth
  if (points.length > 500) {
    const head = points.slice(0, -100);
    const tail = points.slice(-100);
    const stride = Math.ceil(head.length / 300);
    points = head.filter((_, i) => i % stride === 0).concat(tail);
  }
  let lo = opts.min ?? Infinity, hi = opts.max ?? -Infinity;
  if (opts.min == null || opts.max == null) {
    for (const p of points) { if (p.v < lo) lo = p.v; if (p.v > hi) hi = p.v; }
    if (opts.min != null) lo = opts.min;
    if (opts.max != null) hi = opts.max;
  }
  const range = hi - lo || 1;
  // Fixed time window (opts.t0/t1) → the whole line scrolls left smoothly
  // every frame instead of jumping when a new sample arrives
  const t0 = opts.t0 ?? points[0].t;
  const t1 = opts.t1 ?? points[points.length - 1].t;
  const tr = t1 - t0 || 1;
  const X = t => ((t - t0) / tr) * (w - 8) + 4;
  const Y = v => h - 6 - ((v - lo) / range) * (h - 12);

  // Baseline reference line
  if (opts.baseline != null) {
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(4, Y(opts.baseline)); ctx.lineTo(w - 4, Y(opts.baseline));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = opts.color || '#6366f1';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p.t), Y(p.v)) : ctx.lineTo(X(p.t), Y(p.v))));
  ctx.stroke();

  // Event markers (blinks / nods)
  if (opts.events) {
    ctx.fillStyle = opts.eventColor || '#f87171';
    for (const et of opts.events) {
      if (et >= t0 && et <= t1) {
        ctx.beginPath();
        ctx.arc(X(et), h - 5, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

// ===== Smooth render loop =====
// requestAnimationFrame @ display rate (usually 60fps). Per-frame work is
// kept compositor-cheap: two small canvas strokes + transform-only bar
// updates. Text/DOM classes update on a 250ms slow path — numbers don't
// need 60fps, but motion does.
const vizDisp = { probs: [0, 0, 0, 0, 0], perclos: 0, drowsyScore: 0 };
let _lastSlowUpdate = 0;
const _lerp = (cur, target) => cur + (target - cur) * 0.18;

function vizFrame() {
  requestAnimationFrame(vizFrame);
  if (!vizEls || document.hidden) return;
  if (!state.webcamRunning) { vizEls.panel.classList.add('hidden'); return; }
  vizEls.panel.classList.remove('hidden');

  const now = Date.now();

  // Sparklines scroll left continuously (fixed 60s window anchored to now)
  drawSparkline(vizEls.eyeCanvas, vizStream.eye, {
    min: 0, max: 1, color: '#38bdf8', baseline: drowsy.baselineEye,
    events: drowsy.blinks.map(b => b.t), eventColor: '#38bdf8',
    t0: now - VIZ_STREAM_MS, t1: now,
  });
  drawSparkline(vizEls.pitchCanvas, vizStream.pitch, {
    color: '#a78bfa', events: drowsy.nods, eventColor: '#f87171',
    t0: now - VIZ_STREAM_MS, t1: now,
  });

  // Bars glide toward their targets (exponential lerp, transform-only)
  const probs = state.lastProbs;
  if (probs) {
    for (let i = 0; i < vizEls.stateFills.length; i++) {
      vizDisp.probs[i] = _lerp(vizDisp.probs[i], probs[i] || 0);
      vizEls.stateFills[i].style.transform = `scaleX(${vizDisp.probs[i].toFixed(4)})`;
    }
  }
  if (drowsy.features) {
    vizDisp.perclos = _lerp(vizDisp.perclos, Math.min(1, drowsy.features[0] * 2.5));
    vizEls.perclosBar.style.transform = `scaleX(${vizDisp.perclos.toFixed(4)})`;
  }
  if (drowsy.model && drowsy.predSmoothed) {
    vizDisp.drowsyScore = _lerp(vizDisp.drowsyScore, Math.min(1, drowsy.predSmoothed.score));
    vizEls.drowsyBar.style.transform = `scaleX(${vizDisp.drowsyScore.toFixed(4)})`;
  }

  // Slow path: text and class changes, 4x/sec
  if (now - _lastSlowUpdate >= 250) {
    _lastSlowUpdate = now;
    renderVizSlow(probs);
  }
}

function renderVizSlow(probs) {
  setText(vizEls.eyeStat, drowsy.baselineEye
    ? `基準 ${drowsy.baselineEye.toFixed(2)}`
    : 'ベースライン計測中...');
  const curPitch = state.gaze.headDirection?.pitch ?? 0;
  setText(vizEls.pitchStat, `${curPitch.toFixed(0)}°`);

  const f = drowsy.features;
  if (f) {
    const [perclos, blinkRate, meanBlinkDur, longBlinks, yawns, nods, , pitchDrift, , elapsedMin] = f;
    setText(vizEls.perclosVal, (perclos * 100).toFixed(1) + '%');
    setText(vizEls.chipBlink, blinkRate.toFixed(0));
    setText(vizEls.chipDur, (meanBlinkDur * 1000).toFixed(0));
    setText(vizEls.chipLong, String(longBlinks));
    setText(vizEls.chipYawn, String(yawns));
    setText(vizEls.chipNod, String(nods));
    setText(vizEls.chipDrift, pitchDrift.toFixed(1));
    setText(vizEls.chipElapsed, elapsedMin.toFixed(1));

    // Pulse the chip whenever its count ticks up, so "AI noticed the yawn/nod"
    // reads visually without needing narration (mirrors the demo script in
    // DROWSY_GUIDE.md: yawn -> chip increases, nod -> pitch graph marker).
    if (vizEls._prevCounts) {
      if (longBlinks > vizEls._prevCounts.longBlinks) flashEl(vizEls.chipLong.parentElement);
      if (yawns > vizEls._prevCounts.yawns) flashEl(vizEls.chipYawn.parentElement);
      if (nods > vizEls._prevCounts.nods) flashEl(vizEls.chipNod.parentElement);
    }
    vizEls._prevCounts = { longBlinks, yawns, nods };
  }

  if (probs) {
    let top = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
    vizEls.stateBars.forEach((row, i) => {
      row.classList.toggle('viz-top', i === top);
      setText(vizEls.stateVals[i], Math.round(probs[i] * 100) + '%');
    });
  }

  if (drowsy.model && drowsy.predSmoothed) {
    vizEls.drowsyNone.style.display = 'none';
    vizEls.drowsyArea.style.display = '';
    const { probs: dp, score } = drowsy.predSmoothed;
    const col = score > 0.6 ? '#f87171' : score > 0.35 ? '#fbbf24' : '#34d399';
    const label = score > 0.6 ? '眠気サイン継続中!' : score > 0.35 ? 'やや眠そう' : '覚醒';
    setText(vizEls.drowsyScore, (score * 100).toFixed(0));
    if (vizEls._lastDrowsyCol !== col) {
      vizEls._lastDrowsyCol = col;
      vizEls.drowsyScore.parentElement.style.color = col;
      vizEls.drowsyBar.style.background = col;
    }
    setText(vizEls.drowsyLabel, label);
    setText(vizEls.drowsyP0, (dp[0] * 100).toFixed(0));
    setText(vizEls.drowsyP1, (dp[1] * 100).toFixed(0));
    setText(vizEls.drowsyP2, ((dp[2] || 0) * 100).toFixed(0));
  }

  setText(vizEls.rowCount, drowsy.recording
    ? `⏺ ${drowsy.rowsSent}行`
    : (drowsy.rowsSent ? `${drowsy.rowsSent}行保存済` : ''));
}

initVizPanel();
requestAnimationFrame(vizFrame);

// ========== Initialization ==========
async function init() {
  try {
    await initPoseLandmarker();
    await loadML();
    await loadDrowsyModel();
  } catch (e) {
    console.error("Init error:", e);
    showToast("⚠️ 初期化に失敗しました");
  } finally {
    if (loadingOverlay) {
      loadingOverlay.classList.add("fade-out");
      setTimeout(() => loadingOverlay.remove(), 600);
    }
  }
}

init();
