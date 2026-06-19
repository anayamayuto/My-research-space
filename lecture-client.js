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
  emotion: { frown: 0, smile: 0, eyeOpenness: 1 },
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
};

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

      for (const s of shapes) {
        if (s.categoryName === "browDownLeft") browDownL = s.score;
        if (s.categoryName === "browDownRight") browDownR = s.score;
        if (s.categoryName === "mouthSmileLeft") smileL = s.score;
        if (s.categoryName === "mouthSmileRight") smileR = s.score;
        if (s.categoryName === "eyeBlinkLeft") blinkL = s.score;
        if (s.categoryName === "eyeBlinkRight") blinkR = s.score;
      }

      state.emotion.frown = Math.round(Math.max(browDownL, browDownR) * 100) / 100;
      state.emotion.smile = Math.round(Math.max(smileL, smileR) * 100) / 100;
      const avgBlink = (blinkL + blinkR) / 2;
      state.emotion.eyeOpenness = Math.round((1.0 - avgBlink) * 100) / 100;
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
    checkAITriggers();

    // Limit data capture to ~5 times per second (200ms) to avoid massive JSON sizes
    if (Date.now() - lastCaptureTime > 200) {
      lastCaptureTime = Date.now();
      
      // Buffer data point with full AI training structure
      const videoTime = mainVideo && !mainVideo.paused ? mainVideo.currentTime : 0;
      const elapsedSec = state.videoStartTime ? (Date.now() - state.videoStartTime) / 1000 : 0;
      state.interaction.timeSinceAction = Math.round((Date.now() - state.interaction._lastActionTime) / 100) / 10;

      state.dataBuffer.push({
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
      });
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

// ========== VR Space Score Texture ==========
const vrScoreCanvas = document.getElementById("vrScoreCanvas");
const vrScoreCtx = vrScoreCanvas ? vrScoreCanvas.getContext("2d") : null;

function updateVRScoreDisplay() {
  if (!vrScoreCtx) return;
  const c = vrScoreCanvas;
  const ctx = vrScoreCtx;
  ctx.fillStyle = "rgba(10, 14, 26, 0.95)";
  ctx.fillRect(0, 0, c.width, c.height);

  // Title
  ctx.fillStyle = "#94a3b8";
  ctx.font = "bold 36px 'Hiragino Kaku Gothic Pro', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("📊 姿勢 & 視線", c.width / 2, 50);

  // Divider
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(40, 68); ctx.lineTo(c.width - 40, 68); ctx.stroke();

  const s = state.scores;
  const g = state.gaze;
  const gazeLabels = { screen: "スクリーン", scoreBoard: "ボード", other: "その他" };
  const items = [
    { label: "総合スコア", val: s.overall, y: 110 },
    { label: "肩の傾き", val: s.shoulderTilt, y: 180 },
    { label: "頭の前傾", val: s.headForward, y: 250 },
    { label: "上半身の傾き", val: s.bodyLean, y: 320 },
  ];

  for (const item of items) {
    ctx.font = "24px 'Hiragino Kaku Gothic Pro', sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.fillText(item.label, 40, item.y);

    ctx.font = "bold 28px Inter, sans-serif";
    const cls = getScoreClass(item.val);
    ctx.fillStyle = cls === "good" ? "#34d399" : cls === "warn" ? "#fbbf24" : "#f87171";
    ctx.textAlign = "right";
    ctx.fillText(item.val.toString(), c.width - 40, item.y);

    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(40, item.y + 10, c.width - 80, 8);
    ctx.fillStyle = cls === "good" ? "#34d399" : cls === "warn" ? "#fbbf24" : "#f87171";
    ctx.fillRect(40, item.y + 10, (c.width - 80) * item.val / 100, 8);
  }

  // Gaze section
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath(); ctx.moveTo(40, 355); ctx.lineTo(c.width - 40, 355); ctx.stroke();

  ctx.font = "bold 24px 'Hiragino Kaku Gothic Pro', sans-serif";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "left";
  ctx.fillText("👁️ 注目先", 40, 390);
  ctx.font = "bold 28px Inter, sans-serif";
  ctx.fillStyle = g.lookingAt === "screen" ? "#34d399" : "#fbbf24";
  ctx.textAlign = "right";
  ctx.fillText(gazeLabels[g.lookingAt] || "---", c.width - 40, 390);

  ctx.font = "24px 'Hiragino Kaku Gothic Pro', sans-serif";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "left";
  ctx.fillText("安定度", 40, 430);
  ctx.font = "bold 28px Inter, sans-serif";
  ctx.fillStyle = g.gazeStability > 0.7 ? "#34d399" : "#fbbf24";
  ctx.textAlign = "right";
  ctx.fillText(Math.round(g.gazeStability * 100) + "%", c.width - 40, 430);

  ctx.font = "24px 'Hiragino Kaku Gothic Pro', sans-serif";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "left";
  ctx.fillText("頭の向き(Y/P/R)", 40, 470);
  ctx.font = "20px Inter, sans-serif";
  ctx.fillStyle = "#e2e8f0";
  ctx.textAlign = "right";
  ctx.fillText(`${g.headDirection.yaw}° / ${g.headDirection.pitch}° / ${g.headDirection.roll}°`, c.width - 40, 470);

  // Notify A-Frame texture update
  const mat = document.querySelector('[material*="vrScoreCanvas"]');
  if (mat && mat.components.material && mat.components.material.material.map) {
    mat.components.material.material.map.needsUpdate = true;
  }
}

// Periodic VR score update
setInterval(updateVRScoreDisplay, 1000);

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
  // Cooldown reduced from 10s to 2s to allow rapid testing of different patterns
  if (Date.now() - state.aiTriggerTracker.lastMessageTime < 2000) return; 
  
  aiSupporterPanel.classList.remove("hidden");
  aiMessageText.textContent = text;
  aiActionsContainer.innerHTML = "";
  
  if (pattern === 'B' || pattern === 'C') {
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
  
  if (pattern === 'D') {
    setTimeout(setDefaultAIMessage, 5000);
  }
}

function checkAITriggers() {
  if (!state.webcamRunning) return; // Only trigger when tracking is active
  
  const videoActive = state.videoLoaded && !mainVideo.paused;
  
  const now = Date.now();
  const tracker = state.aiTriggerTracker;
  const emo = state.emotion;
  const hands = state.hands;
  const gaze = state.gaze;
  const scores = state.scores;
  
  // Pattern A: Confusion (Touch face OR frown > 0.2 for 2s)
  if (hands.isTouchingFace || emo.frown > 0.2) {
    if (!tracker.confusionStart) tracker.confusionStart = now;
    else if (now - tracker.confusionStart > 2000) {
      showAIMessage('A', "少し難しいですか？この部分は重要なので、もう一度見直すのもおすすめです。", [
        { label: "⏪ 10秒戻る", onClick: () => { mainVideo.currentTime = Math.max(0, mainVideo.currentTime - 10); setDefaultAIMessage(); tracker.confusionStart = null; } }
      ]);
      tracker.confusionStart = null;
    }
  } else {
    tracker.confusionStart = null;
  }
  
  // Pattern B: Distraction (Not looking at screen or eyes closed for 5s)
  // Relaxed conditions: Head turned > 25 degrees, or eyes almost closed (< 0.4)
  const isHeadTurned = gaze.headDirection && (Math.abs(gaze.headDirection.yaw) > 25 || Math.abs(gaze.headDirection.pitch) > 25);
  const isDistracted = emo.eyeOpenness < 0.4 || isHeadTurned;
  
  if (videoActive && isDistracted) {
    if (!tracker.distractionStart) tracker.distractionStart = now;
    else if (now - tracker.distractionStart > 5000) {
      mainVideo.pause();
      showAIMessage('B', "少し疲れが見えますね。深呼吸してリフレッシュしましょう！", [
        { label: "✅ 了解", onClick: () => { setDefaultAIMessage(); tracker.distractionStart = null; mainVideo.play(); } }
      ]);
      tracker.distractionStart = null;
    }
  } else if (videoActive) {
    tracker.distractionStart = null;
  }
  
  // Pattern C: Bad Posture (Score < 60 for 5s)
  if (scores.overall < 60) {
    if (!tracker.badPostureStart) tracker.badPostureStart = now;
    else if (now - tracker.badPostureStart > 5000) {
      showAIMessage('C', "姿勢が悪くなっています。画面に近づきすぎていませんか？少し背筋を伸ばしてみましょう。");
      tracker.badPostureStart = null;
    }
  } else {
    tracker.badPostureStart = null;
  }
  
  // Pattern D: Flow (Score > 70 and looking straight for 15s)
  // Changed from >80 because score fluctuates rapidly, making 15s hold impossible if it drops to 79 for a frame
  if (scores.overall >= 70 && !isHeadTurned) {
    if (!tracker.flowStart) tracker.flowStart = now;
    else if (now - tracker.flowStart > 15000) {
      showAIMessage('D', "素晴らしい集中力ですね！その調子です！");
      tracker.flowStart = null;
    }
  } else {
    tracker.flowStart = null;
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

// ========== Initialization ==========
async function init() {
  try {
    await initPoseLandmarker();
  } catch (e) {
    console.error("Init error:", e);
    showToast("⚠️ モデルの初期化に失敗しました");
  } finally {
    if (loadingOverlay) {
      loadingOverlay.classList.add("fade-out");
      setTimeout(() => loadingOverlay.remove(), 600);
    }
  }
}

init();
