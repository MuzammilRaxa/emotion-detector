// "use client";

// import { useEffect, useRef, useState } from "react";
// import * as faceapi from "face-api.js";
// import Webcam from "react-webcam";
// import * as tf from "@tensorflow/tfjs"; // core
// import "@tensorflow/tfjs-backend-webgl"; // webgl backend (optional to import)

// export default function EmotionTracker() {
//   const webcamRef = useRef(null);
//   const [status, setStatus] = useState("Idle");
//   const [emotionList, setEmotionList] = useState([]);
//   const detectionIntervalRef = useRef(null);

//   useEffect(() => {
//     let cancelled = false;

//     const prepare = async () => {
//       try {
//         setStatus("Checking TF backend support...");
//         // Try WebGL first, fallback to CPU if it fails
//         try {
//           await tf.setBackend("webgl");
//           await tf.ready();
//           console.log("TF backend ready:", tf.getBackend());
//         } catch (err) {
//           console.warn("WebGL backend failed, falling back to cpu:", err);
//           try {
//             await tf.setBackend("cpu");
//             await tf.ready();
//             console.log("TF CPU backend ready:", tf.getBackend());
//           } catch (err2) {
//             console.error("Failed to set any TF backend:", err2);
//             setStatus("Error: no TF backend available");
//             return;
//           }
//         }

//         setStatus("Loading face-api models...");
//         const MODEL_URL = "/models";

//         // IMPORTANT: check network tab for these GETs to succeed (200)
//         await Promise.all([
//           faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
//           faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
//         ]);

//         if (cancelled) return;

//         setStatus("Models loaded. Waiting for webcam...");

//         // Wait until webcam video is ready and playing
//         await waitForVideoToPlay(webcamRef);

//         if (cancelled) return;

//         setStatus("Starting detection...");
//         startDetectionLoop();
//       } catch (err) {
//         console.error("Initialization error:", err);
//         setStatus("Initialization error, check console");
//       }
//     };

//     prepare();

//     return () => {
//       cancelled = true;
//       stopDetectionLoop();
//     };
//   }, []);

//   // Wait until the webcam video element exists and is playing
//   const waitForVideoToPlay = (webcamRef, timeout = 10000) => {
//     return new Promise((resolve, reject) => {
//       const start = Date.now();
//       const check = () => {
//         const video = webcamRef.current?.video;
//         if (video) {
//           // readyState >= 2 means HAVE_CURRENT_DATA; play to ensure frames flowing
//           if (video.readyState >= 2) {
//             // ensure playing
//             if (video.paused) {
//               const playPromise = video.play();
//               if (playPromise && typeof playPromise.then === "function") {
//                 playPromise
//                   .then(() => {
//                     console.log("Video started playing.");
//                     resolve();
//                   })
//                   .catch((err) => {
//                     console.warn("Video.play() rejected:", err);
//                     // still resolve because many browsers require user gesture (webcam allowed)
//                     resolve();
//                   });
//               } else {
//                 resolve();
//               }
//             } else {
//               resolve();
//             }
//             return;
//           }
//         }
//         if (Date.now() - start > timeout) {
//           reject(new Error("Timed out waiting for video to be ready"));
//           return;
//         }
//         requestAnimationFrame(check);
//       };
//       check();
//     });
//   };

//   // Start interval-based detection
//   const startDetectionLoop = () => {
//     stopDetectionLoop(); // ensure no duplicate loop
//     detectionIntervalRef.current = setInterval(async () => {
//       try {
//         const video = webcamRef.current?.video;
//         if (!video) {
//           // no video yet
//           return;
//         }

//         // Do detection only when playing and has frames
//         if (video.readyState < 2) return;

//         // Use tinyFaceDetector options (tune inputSize / scoreThreshold if needed)
//         const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });

//         const detections = await faceapi
//           .detectAllFaces(video, options)
//           .withFaceExpressions();

//         if (detections && detections.length > 0) {
//           const mainFace = detections[0].expressions;

//           // **Emotion filtering & validation:**
//           const emotions = Object.keys(mainFace)
//             .filter((key) => mainFace[key] > 0.5) // Filter out low-confidence emotions
//             .sort((a, b) => mainFace[b] - mainFace[a]); // Sort emotions by confidence

//           // Skip if no strong emotion detected
//           if (emotions.length === 0) return;

//           const emotion = emotions[0]; // Get the most confident emotion
//           const confidence = mainFace[emotion];

//           const newEntry = {
//             timestamp: new Date().toLocaleTimeString(),
//             emotion,
//             confidence: confidence.toFixed(2),
//           };

//           setEmotionList((prev) => [newEntry, ...prev.slice(0, 199)]); // keep last 200
//         }
//       } catch (err) {
//         console.error("Detection error:", err);
//         // If WebGL keeps failing at runtime, switch to CPU backend automatically
//         if (err && /WebGL|webgl/i.test(String(err))) {
//           try {
//             console.warn("Switching to CPU backend due to runtime WebGL error...");
//             await tf.setBackend("cpu");
//             await tf.ready();
//             console.log("Switched to backend:", tf.getBackend());
//           } catch (be) {
//             console.error("Failed to switch backend:", be);
//           }
//         }
//       }
//     }, 200); // run every 200ms (5 FPS) — change to 100ms for higher frequency
//   };

//   const stopDetectionLoop = () => {
//     if (detectionIntervalRef.current) {
//       clearInterval(detectionIntervalRef.current);
//       detectionIntervalRef.current = null;
//     }
//   };

//   // CSV generation and download (same as before)
//   const generateCSV = () => {
//     const headers = ["Timestamp", "Emotion", "Confidence"];
//     const rows = emotionList.map((e) => [e.timestamp, e.emotion, e.confidence]);
//     const csvContent = [headers, ...rows].map((r) => r.join(",")).join("\n");
//     return csvContent;
//   };

//   const downloadCSV = () => {
//     const csv = generateCSV();
//     const blob = new Blob([csv], { type: "text/csv" });
//     const url = window.URL.createObjectURL(blob);
//     const a = document.createElement("a");
//     a.href = url;
//     a.download = `emotion_log_${Date.now()}.csv`;
//     a.click();
//     window.URL.revokeObjectURL(url);
//   };

//   return (
//     <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
//       <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
//         <h2 className="text-2xl font-bold mb-2 text-gray-700">Emotion Tracker</h2>
//         <p className="text-sm text-gray-500 mb-3">{status}</p>
//         <Webcam
//           ref={webcamRef}
//           width={420}
//           height={300}
//           className="rounded-lg border border-gray-300"
//           audio={false}
//           videoConstraints={{ facingMode: "user" }}
//         />
//         <button
//           onClick={downloadCSV}
//           className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
//         >
//           📥 Download CSV
//         </button>
//       </div>

//       <div className="bg-white w-full md:w-80 h-[400px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
//         <h3 className="text-lg font-semibold mb-3 text-gray-700 flex items-center gap-2">
//           🧠 Emotion Log
//         </h3>

//         {emotionList.length === 0 ? (
//           <p className="text-gray-400 text-sm text-center mt-10">
//             No emotions detected yet...
//           </p>
//         ) : (
//           <ul className="space-y-2">
//             {emotionList.map((e, index) => (
//               <li
//                 key={index}
//                 className="flex justify-between items-center bg-gray-50 hover:bg-blue-50 border border-gray-200 rounded-lg px-3 py-2 transition"
//               >
//                 <div>
//                   <p className="font-medium capitalize text-gray-800">{e.emotion}</p>
//                   <p className="text-xs text-gray-500">{e.timestamp}</p>
//                 </div>
//                 <span className="text-sm font-semibold text-blue-600">{e.confidence}</span>
//               </li>
//             ))}
//           </ul>
//         )}
//       </div>
//     </div>
//   );
// }
