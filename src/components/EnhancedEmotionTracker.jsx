// "use client";

// import React, { useEffect, useRef, useState } from "react";
// import * as faceapi from "face-api.js";
// import Webcam from "react-webcam";

// import { motion, AnimatePresence } from "framer-motion";
// import * as tf from "@tensorflow/tfjs";
// import "@tensorflow/tfjs-backend-wasm"; // include WASM backend

// export default function EnhancedEmotionTracker() {
//   const MODEL_URL = "/models"; // ensure models are served here
//   const webcamRef = useRef(null);
//   const overlayRef = useRef(null);

//   const rafRef = useRef(null);

//   const [status, setStatus] = useState("Idle");
//   const [modelsLoaded, setModelsLoaded] = useState(false); // ✅ fixed: use modelsLoaded, not "loaded"
//   const [running, setRunning] = useState(false);
//   const [darkMode, setDarkMode] = useState(false);
//   const [emotionEvents, setEmotionEvents] = useState([]);
//   const [faceTracks, setFaceTracks] = useState({});
//   const [stats, setStats] = useState({ fps: 0, faces: 0, topEmotion: "—" });
//   const [toasts, setToasts] = useState([]);

//   const FPS = 6;
//   const STABILITY_COUNT = 3;
//   const STABILITY_WINDOW = 5;

//   const pushToast = (text, type = "info") => {
//     const id = Math.random().toString(36).slice(2, 9);
//     setToasts((t) => [...t, { id, text, type }]);
//     setTimeout(() => {
//       setToasts((t) => t.filter((x) => x.id !== id));
//     }, 3500);
//   };

//   // ✅ Fixed model loader
//   async function loadModels() {
//     try {
//       // Initialize TensorFlow backend properly
//       await tf.setBackend("wasm").catch(() => tf.setBackend("webgl"));
//       await tf.ready();
//       console.log("✅ TensorFlow backend initialized:", tf.getBackend());

//       // Load models sequentially after TF ready
//       await Promise.all([faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL), faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)]);

//       console.log("✅ FaceAPI models loaded successfully");
//       setModelsLoaded(true);
//       setStatus("Models Loaded");
//       pushToast("Models loaded successfully!", "success");
//     } catch (err) {
//       console.error("❌ Model load error:", err);
//       pushToast("Model loading failed. Check console.", "error");
//     }
//   }

//   useEffect(() => {
//     loadModels();
//   }, []);

//   useEffect(() => {
//     // update stats topEmotion whenever emotionEvents updates
//     if (emotionEvents.length === 0) return;
//     const counts = {};
//     emotionEvents.slice(0, 500).forEach((e) => {
//       counts[e.emotion] = (counts[e.emotion] || 0) + 1;
//     });
//     const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "—";
//     setStats((s) => ({ ...s, topEmotion: top }));
//     // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, [emotionEvents.length]);

//   const start = () => {
//     if (!modelsLoaded) {
//       pushToast("Models are not yet loaded.");
//       return;
//     }
//     setRunning(true);
//     setStatus("Running");
//     runDetectionLoop();
//     pushToast("Detection started", "success");
//   };

//   const stop = () => {
//     setRunning(false);
//     setStatus("Stopped");
//     if (rafRef.current) cancelAnimationFrame(rafRef.current);
//   };

//   const reconnect = () => {
//     // client-only: simply restart detection
//     stop();
//     setTimeout(() => start(), 300);
//     pushToast("Reconnecting...");
//   };

//   // draw overlay boxes
//   const drawDetections = (detections) => {
//     const canvas = overlayRef.current;
//     const video = webcamRef.current?.video;
//     if (video && video.readyState >= 2) {
//       console.log("Webcam feed is ready");
//       // Continue with face-api detection
//     } else {
//       console.error("Webcam feed not ready");
//     }
//     if (!canvas || !video) return;
//     const ctx = canvas.getContext("2d");
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;

//     ctx.clearRect(0, 0, canvas.width, canvas.height);
//     detections.forEach((d) => {
//       const box = d.detection.box;
//       const x = box.x;
//       const y = box.y;
//       const w = box.width;
//       const h = box.height;

//       // bounding box
//       ctx.beginPath();
//       ctx.lineWidth = 2;
//       ctx.strokeStyle = "rgba(34,197,94,0.95)"; // green
//       ctx.rect(x, y, w, h);
//       ctx.stroke();

//       // label + emotion
//       const emotion = getDominantEmotion(d.expressions);
//       const conf = (d.expressions[emotion] || 0) * 100;
//       const label = `${emotion} ${conf.toFixed(0)}%`;

//       ctx.fillStyle = "rgba(0,0,0,0.6)";
//       ctx.fillRect(x, y - 22, ctx.measureText(label).width + 14 || w, 22);
//       ctx.fillStyle = "#fff";
//       ctx.font = "14px sans-serif";
//       ctx.fillText(label, x + 6, y - 6);

//       // small confidence ring (draw at top-right of box)
//       const cx = x + w - 14;
//       const cy = y + 14;
//       const radius = 12;
//       ctx.beginPath();
//       ctx.lineWidth = 3;
//       ctx.strokeStyle = "rgba(59,130,246,0.9)"; // blue
//       ctx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + (conf / 100) * Math.PI * 2);
//       ctx.stroke();
//     });
//   };

//   const getDominantEmotion = (expressions) => {
//     if (!expressions) return "neutral";
//     const entries = Object.entries(expressions);
//     entries.sort((a, b) => b[1] - a[1]);
//     return entries[0]?.[0] || "neutral";
//   };

//   // track smoothing & stability
//   const updateTracksWithDetections = (detections) => {
//     const updated = { ...faceTracks };
//     const nowStr = new Date().toLocaleTimeString();
//     const newEvents = [];

//     detections.forEach((d, idx) => {
//       // Use bounding box center to compute a pseudo-id by spatial hashing for stable demo
//       const box = d.detection.box;
//       const centerX = Math.round(box.x + box.width / 2);
//       const centerY = Math.round(box.y + box.height / 2);
//       const spatialId = `s_${Math.round(centerX / 50)}_${Math.round(centerY / 50)}`;
//       const faceId = spatialId; // in absence of a reid model, spatial hashing is used

//       if (!updated[faceId]) {
//         updated[faceId] = {
//           id: faceId,
//           firstSeen: nowStr,
//           lastSeen: nowStr,
//           buffer: [], // recent detected emotions
//           confirmedEmotion: null,
//           confirmedSince: null,
//           detectionCount: 0,
//           bbox: [box.x, box.y, box.width, box.height],
//         };
//         pushToast(`New face: ${faceId}`, "info");
//       }

//       const track = updated[faceId];
//       track.lastSeen = nowStr;
//       track.detectionCount++;
//       track.bbox = [box.x, box.y, box.width, box.height];

//       const dominant = getDominantEmotion(d.expressions);
//       const conf = d.expressions[dominant] || 0;

//       // push into buffer
//       track.buffer.unshift({ emotion: dominant, conf, t: Date.now() });
//       if (track.buffer.length > STABILITY_WINDOW) track.buffer.pop();

//       // compute most frequent in buffer
//       const freq = {};
//       track.buffer.forEach((b) => (freq[b.emotion] = (freq[b.emotion] || 0) + 1));
//       const most = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
//       if (most) {
//         const [mostEmotion, count] = most;
//         if (count >= STABILITY_COUNT && track.confirmedEmotion !== mostEmotion) {
//           // confirmed change
//           track.confirmedEmotion = mostEmotion;
//           track.confirmedSince = nowStr;
//           // push an event
//           const event = {
//             timestamp: nowStr,
//             faceId: `user_${track.id}`,
//             emotion: mostEmotion,
//             confidence: (track.buffer[0]?.conf || 0).toFixed(3),
//             bbox: track.bbox,
//             isNewFace: track.detectionCount <= 3,
//             emotionChanged: true,
//           };
//           newEvents.push(event);
//           pushToast(`Emotion changed for ${track.id}: ${mostEmotion}`, "success");
//         }
//       }
//     });

//     // cleanup stale tracks (not seen for N seconds)
//     const STALE_MS = 12_000;
//     Object.keys(updated).forEach((k) => {
//       const t = updated[k];
//       if (Date.now() - (new Date(`1970-01-01T${t.lastSeen}`)?.getTime() || Date.now()) > STALE_MS) {
//         // note: lastSeen is a time string; avoid removing aggressively in demo — keep safer approach
//       }
//     });

//     // merge new events
//     if (newEvents.length > 0) {
//       setEmotionEvents((prev) => [...newEvents, ...prev].slice(0, 1000));
//     }

//     setFaceTracks(updated);
//     setStats((s) => ({ ...s, faces: Object.keys(updated).length }));
//   };

//   // detection loop
//   const runDetectionLoop = async () => {
//     let lastTime = performance.now();
//     let frames = 0;
//     let lastFPSUpdate = performance.now();

//     const loop = async () => {
//       if (!running) return;
//       const now = performance.now();
//       const elapsed = now - lastTime;

//       // throttle by FPS
//       if (elapsed >= 1000 / FPS) {
//         console.log("Detection detections:");
//         lastTime = now;
//         frames++;

//         try {
//           const video = webcamRef.current?.video;
//           if (video && video.readyState >= 2) {
//             console.log("Detection starting...");
//             const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
//             const detections = await faceapi.detectAllFaces(video, options).withFaceExpressions().withFaceLandmarks();
//             console.log("Detection results:", detections);

//             drawDetections(detections);
//             updateTracksWithDetections(detections);
//           } else {
//             console.error("Video feed not ready");
//           }
//         } catch (err) {
//           console.error("Detection error:", err);
//         }
//       }

//       // update FPS every second
//       if (now - lastFPSUpdate > 1000) {
//         setStats((s) => ({ ...s, fps: frames }));
//         frames = 0;
//         lastFPSUpdate = now;
//       }

//       rafRef.current = requestAnimationFrame(loop);
//     };

//     rafRef.current = requestAnimationFrame(loop);
//   };

//   // CSV export
//   const generateCSV = () => {
//     const headers = ["Timestamp", "Face ID", "Emotion", "Confidence", "BBoxX", "BBoxY", "BBoxW", "BBoxH", "EmotionChanged", "NewFace"];
//     const rows = emotionEvents.map((e) => [e.timestamp, e.faceId, e.emotion, e.confidence, ...(e.bbox || []), e.emotionChanged ? "YES" : "NO", e.isNewFace ? "YES" : "NO"]);
//     const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
//     return csv;
//   };

//   const downloadCSV = () => {
//     const csv = generateCSV();
//     const blob = new Blob([csv], { type: "text/csv" });
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement("a");
//     a.href = url;
//     a.download = `emotion_events_${Date.now()}.csv`;
//     a.click();
//     URL.revokeObjectURL(url);
//     pushToast("CSV exported", "success");
//   };

//   // small helper to format time nicely
//   const niceTime = (ts) => ts;

//   return (
//     <div className={darkMode ? "dark" : ""}>
//       <div className="min-h-screen bg-gradient-to-b from-gray-100 to-white dark:from-gray-900 dark:to-gray-800 p-6">
//         <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
//           {/* Main camera card */}
//           <div className="md:col-span-2 bg-white dark:bg-gray-900/40 backdrop-blur-md rounded-2xl shadow p-4 relative">
//             {/* Top bar */}
//             <div className="flex items-center justify-between mb-3">
//               <div className="flex items-center gap-3">
//                 <div className={`w-3 h-3 rounded-full ${running ? "bg-green-400" : "bg-red-400"}`}></div>
//                 <div>
//                   <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">Enhanced Emotion Tracker</div>
//                   <div className="text-xs text-gray-500 dark:text-gray-400">
//                     {status} • FPS: {stats.fps} • Faces: {stats.faces}
//                   </div>
//                 </div>
//               </div>

//               <div className="flex items-center gap-2">
//                 <button onClick={() => setDarkMode((d) => !d)} className="px-3 py-1 rounded-md bg-gray-100 dark:bg-gray-800 text-xs">
//                   {darkMode ? "Light" : "Dark"}
//                 </button>
//                 <button onClick={reconnect} className="px-3 py-1 rounded-md bg-yellow-100 text-xs">
//                   Reconnect
//                 </button>
//                 {!running ? (
//                   <button onClick={start} className="px-4 py-2 rounded-md bg-blue-600 text-white">
//                     Start
//                   </button>
//                 ) : (
//                   <button onClick={stop} className="px-4 py-2 rounded-md bg-red-500 text-white">
//                     Stop
//                   </button>
//                 )}
//               </div>
//             </div>

//             <div className="relative w-full overflow-hidden rounded-lg">
//               <Webcam ref={webcamRef} audio={false} className="w-full h-auto rounded-lg" videoConstraints={{ facingMode: "user", width: 720, height: 540 }} />

//               <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />

//               {/* small stats card */}
//               <div className="absolute top-4 left-4 bg-white/70 dark:bg-black/50 rounded-xl p-2 text-xs">
//                 <div className="font-semibold">
//                   Top: <span className="capitalize">{stats.topEmotion}</span>
//                 </div>
//               </div>
//             </div>

//             {/* bottom timeline preview */}
//             <div className="mt-4">
//               <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2">Recent Events</h4>
//               <div className="flex gap-2 overflow-x-auto pb-2">
//                 {emotionEvents.slice(0, 12).map((e, i) => (
//                   <div key={i} className="min-w-[180px] bg-gray-50 dark:bg-gray-800 rounded-lg p-2 text-xs border">
//                     <div className="flex justify-between items-center">
//                       <div className="font-medium capitalize">{e.emotion}</div>
//                       <div className="text-blue-600 font-semibold">{e.confidence}</div>
//                     </div>
//                     <div className="text-gray-500 mt-1 text-[11px]">
//                       {e.faceId} • {e.timestamp}
//                     </div>
//                     <div className="mt-2 text-[11px] text-gray-600 dark:text-gray-300">
//                       {e.isNewFace ? "New" : ""} {e.emotionChanged ? "Changed" : ""}
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>

//           {/* Right column: detailed log & controls */}
//           <div className="bg-white dark:bg-gray-900/40 rounded-2xl shadow p-4">
//             <div className="flex items-center justify-between mb-3">
//               <div>
//                 <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">Emotion Log</div>
//                 <div className="text-xs text-gray-500 dark:text-gray-400">Last {Math.min(emotionEvents.length, 200)} events</div>
//               </div>
//               <div className="flex gap-2">
//                 <button onClick={downloadCSV} className="px-3 py-1 bg-blue-600 text-white rounded text-xs">
//                   Export CSV
//                 </button>
//               </div>
//             </div>

//             <div style={{ maxHeight: 420 }} className="overflow-y-auto">
//               <AnimatePresence initial={false}>
//                 {emotionEvents.slice(0, 200).map((e, i) => (
//                   <motion.div
//                     key={e.timestamp + i}
//                     initial={{ opacity: 0, y: 8 }}
//                     animate={{ opacity: 1, y: 0 }}
//                     exit={{ opacity: 0, y: -8 }}
//                     transition={{ duration: 0.18 }}
//                     className={`flex justify-between items-start p-2 rounded-lg mb-2 border ${e.emotionChanged ? "border-green-300" : "border-gray-200"} bg-white/50 dark:bg-gray-800/50`}>
//                     <div>
//                       <div className="font-medium capitalize">{e.emotion}</div>
//                       <div className="text-[11px] text-gray-500">
//                         {e.faceId} • {niceTime(e.timestamp)}
//                       </div>
//                     </div>
//                     <div className="text-right">
//                       <div className="font-semibold text-blue-600">{e.confidence}</div>
//                       <div className="text-[11px] text-gray-500">{e.emotionChanged ? "Changed" : ""}</div>
//                     </div>
//                   </motion.div>
//                 ))}
//               </AnimatePresence>
//             </div>

//             <div className="mt-4">
//               <h5 className="text-xs text-gray-500 mb-2">Active Faces</h5>
//               <div className="space-y-2 text-xs">
//                 {Object.values(faceTracks).map((t) => (
//                   <div key={t.id} className="flex justify-between items-center bg-gray-50 dark:bg-gray-800/50 rounded p-2">
//                     <div>
//                       <div className="font-medium">{`user_${t.id}`}</div>
//                       <div className="text-[11px] text-gray-500">
//                         Seen: {t.firstSeen} • Last: {t.lastSeen}
//                       </div>
//                     </div>
//                     <div className="text-right">
//                       <div className="text-sm font-semibold capitalize">{t.confirmedEmotion || "—"}</div>
//                       <div className="text-[11px] text-gray-500">{t.detectionCount} frames</div>
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>
//         </div>

//         {/* toasts */}
//         <div className="fixed right-6 bottom-6 z-50 flex flex-col gap-2">
//           {toasts.map((t) => (
//             <div key={t.id} className="min-w-[220px] bg-black/80 text-white p-3 rounded-md shadow">
//               <div className="text-sm">{t.text}</div>
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// }