"use client";
// #V1 working  on PRODUCTION Emotion Detection Server with Advanced Face Tracking
import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

export default function EmotionTrackerPython() {
    const webcamRef = useRef(null);
    const [status, setStatus] = useState("Idle");
    const [emotionList, setEmotionList] = useState([]);
    const [connected, setConnected] = useState(false);
    const [faceTracks, setFaceTracks] = useState({});
    const wsRef = useRef(null);
    const animationRef = useRef(null);
    const frameCounterRef = useRef(0);

    // Utility functions
    const checkEmotionChange = (previousEmotions, currentDetection) => {
        if (!previousEmotions || previousEmotions.length === 0) {
            return false;
        }

        // Find the most recent emotion for this face
        const lastEmotionForFace = previousEmotions.find(
            emotion => emotion.faceId === currentDetection.face_id
        );

        if (!lastEmotionForFace) {
            return false;
        }

        return lastEmotionForFace.emotion !== currentDetection.emotion;
    };

    const checkNewFace = (previousEmotions, currentFaceId) => {
        if (!previousEmotions || previousEmotions.length === 0) {
            return true;
        }

        return !previousEmotions.some(emotion => emotion.faceId === currentFaceId);
    };

    useEffect(() => {
        connectWebSocket();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, []);

    const connectWebSocket = () => {
        setStatus("Connecting to AI server...");

        try {
            const ws = new WebSocket("ws://localhost:8765");
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("✅ WebSocket connected successfully");
                setConnected(true);
                setStatus("Connected! Starting detection...");
                startFrameCapture();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'detection_results') {
                        const newFaceTracks = {};

                        data.detections.forEach(detection => {
                            const faceId = detection.face_id;

                            newFaceTracks[faceId] = {
                                id: faceId,
                                personName: detection.person_name || `Person_${detection.face_id}`,
                                bbox: detection.bbox || [0, 0, 0, 0],
                                currentEmotion: detection.emotion || 'neutral',
                                confidence: detection.confidence || 0.8,
                                gender: detection.gender || 'unknown',
                                genderConfidence: detection.gender_confidence || 0.5,
                                detectionCount: detection.track_count || 1,
                                lastSeen: Date.now()
                            };
                        });

                        setFaceTracks(newFaceTracks);

                        // Add to emotion list
                        if (data.detections.length > 0) {
                            setEmotionList(prev => {
                                const newEntries = data.detections.map(detection => ({
                                    personName: detection.person_name || `Person_${detection.face_id}`,
                                    faceId: detection.face_id,
                                    emotion: detection.emotion || 'neutral',
                                    confidence: detection.confidence || 0.8,
                                    gender: detection.gender || 'unknown',
                                    genderConfidence: detection.gender_confidence || 0.5,
                                    bbox: detection.bbox || [],
                                    timestamp: detection.timestamp || new Date().toISOString(),
                                    track_count: detection.track_count || 1,
                                    similarity_score: detection.similarity_score || 0,
                                    emotionChanged: checkEmotionChange(prev, detection),
                                    isNewFace: checkNewFace(prev, detection.face_id)
                                }));

                                return [...newEntries, ...prev].slice(0, 50);
                            });
                        }
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                }
            };

            ws.onclose = (event) => {
                console.log("📞 WebSocket disconnected:", event.code, event.reason);
                setConnected(false);
                setStatus(`Disconnected (${event.code}). Retrying in 3s...`);

                // Clear any existing timeouts to prevent multiple retries
                setTimeout(() => {
                    if (!connected) {
                        connectWebSocket();
                    }
                }, 3000);
            };

            ws.onerror = (error) => {
                console.log("❌ WebSocket error:", error);
                setStatus("Connection error - Check if Python server is running");
                setConnected(false);
            };
        } catch (error) {
            console.log("❌ Failed to create WebSocket:", error);
            setStatus("Failed to connect - Retrying...");
            setTimeout(connectWebSocket, 3000);
        }
    };

    const startFrameCapture = () => {
        let lastSent = 0;
        const FPS = 10; // Reduced to 3 FPS to avoid overwhelming the backend

        const captureFrame = () => {
            // Only capture if connected and webcam is ready
            if (!connected || !webcamRef.current?.video || wsRef.current?.readyState !== WebSocket.OPEN) {
                animationRef.current = requestAnimationFrame(captureFrame);
                return;
            }

            const now = Date.now();
            if (now - lastSent < 1000 / FPS) {
                animationRef.current = requestAnimationFrame(captureFrame);
                return;
            }

            const video = webcamRef.current.video;

            // Check if video is ready and has data
            if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Reduce quality to minimize data size
                    const frameData = canvas.toDataURL('image/jpeg', 0.7);
                    frameCounterRef.current++;

                    // Send frame to backend
                    if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({
                            type: 'frame',
                            frame: frameData,
                            timestamp: now,
                            frameId: frameCounterRef.current
                        }));
                    }

                    lastSent = now;
                } catch (error) {
                    console.log("❌ Error capturing frame:", error);
                }
            }

            animationRef.current = requestAnimationFrame(captureFrame);
        };

        captureFrame();
    };

    // const processDetections = (detections) => {
    //     if (!detections || detections.length === 0) {
    //         return;
    //     }

    //     const newEntries = [];
    //     const updatedTracks = { ...faceTracks };

    //     detections.forEach(detection => {
    //         const { face_id, emotion, confidence, bbox, timestamp, track_count, confirmed } = detection;

    //         // Only process confirmed tracks with valid emotions
    //         if (!confirmed || !emotion || confidence < 0.3) {
    //             return;
    //         }

    //         // Update face tracks
    //         if (!updatedTracks[face_id]) {
    //             updatedTracks[face_id] = {
    //                 id: face_id,
    //                 firstSeen: new Date().toLocaleTimeString(),
    //                 emotionHistory: [],
    //                 currentEmotion: emotion,
    //                 detectionCount: 0,
    //                 lastSeen: new Date().toLocaleTimeString(),
    //                 trackCount: track_count,
    //                 confirmed: confirmed
    //             };
    //         }

    //         const track = updatedTracks[face_id];
    //         track.detectionCount++;
    //         track.currentEmotion = emotion;
    //         track.lastSeen = new Date().toLocaleTimeString();
    //         track.trackCount = track_count;

    //         // Only add entry if emotion changed OR it's been more than 3 seconds
    //         const lastEmotion = track.emotionHistory[0]?.emotion;
    //         const emotionChanged = !lastEmotion || lastEmotion !== emotion;
    //         const timeSinceLastEntry = track.emotionHistory[0] ?
    //             (new Date() - new Date(track.emotionHistory[0].timestamp)) / 1000 : 999;

    //         // More conservative: only log significant changes or every 3+ seconds
    //         if ((emotionChanged && confidence > 0.5) || timeSinceLastEntry > 3) {
    //             const newEntry = {
    //                 timestamp: new Date().toLocaleTimeString(),
    //                 emotion,
    //                 confidence: typeof confidence === 'number' ? confidence.toFixed(3) : confidence,
    //                 faceId: `user_${face_id}`,
    //                 bbox: bbox || [],
    //                 emotionChanged,
    //                 isNewFace: track.detectionCount === 1,
    //                 trackCount: track_count,
    //                 confirmed: confirmed
    //             };

    //             newEntries.push(newEntry);
    //             track.emotionHistory.unshift({
    //                 emotion,
    //                 confidence: typeof confidence === 'number' ? confidence : parseFloat(confidence),
    //                 timestamp: new Date().toLocaleTimeString()
    //             });

    //             // Keep only last 15 emotions per face
    //             if (track.emotionHistory.length > 15) {
    //                 track.emotionHistory = track.emotionHistory.slice(0, 15);
    //             }

    //             console.log(`✅ Track ${face_id} (${track_count} frames): ${emotion} (${confidence})`);
    //         }
    //     });

    //     // Update state with new entries
    //     if (newEntries.length > 0) {
    //         setEmotionList(prev => [...newEntries, ...prev.slice(0, 150)]); // Keep fewer entries
    //     }
    //     setFaceTracks(updatedTracks);
    // };

    const generateEnhancedCSV = () => {
        const headers = [
            "Timestamp", "Person Name", "Face ID", "Emotion", "Confidence",
            "Gender", "Gender Confidence", "Bounding Box", "Emotion Changed",
            "New Face", "Track Count", "Similarity Score"
        ];

        const rows = emotionList.map(e => [
            e.timestamp,
            e.personName || `Person_${e.faceId}`,
            e.faceId,
            e.emotion,
            e.confidence,
            e.gender || 'unknown',
            e.genderConfidence || 'N/A',
            e.bbox && e.bbox.length === 4 ? `[${e.bbox.join(',')}]` : '[]',
            e.emotionChanged ? 'YES' : 'NO',
            e.isNewFace ? 'YES' : 'NO',
            e.track_count || '0',
            e.similarity_score || 'N/A'
        ]);

        const csvContent = [headers, ...rows].map(r => r.join(",")).join("\n");
        return csvContent;
    };

    const downloadCSV = () => {
        if (emotionList.length === 0) {
            alert("No emotion data to download yet!");
            return;
        }

        const csv = generateEnhancedCSV();
        const blob = new Blob([csv], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `emotion_log_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    };

    const getStatusColor = () => {
        if (!connected) return "text-red-500";
        return "text-green-500";
    };

    const getStatusEmoji = () => {
        if (!connected) return "❌";
        return "✅";
    };

    return (
        <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
            {/* Left Panel - Webcam & Controls */}
            <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
                <h2 className="text-2xl font-bold mb-2 text-gray-700">AI Emotion & Gender Tracker</h2>
                <p className={`text-sm mb-3 font-medium ${getStatusColor()}`}>
                    {getStatusEmoji()} {status}
                </p>

                {/* Webcam with face overlay */}
                <div className="relative">
                    <Webcam
                        ref={webcamRef}
                        width={480}
                        height={360}
                        className="rounded-lg border-2 border-gray-300"
                        audio={false}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{
                            facingMode: "user",
                            width: 480,
                            height: 360,
                            frameRate: { ideal: 10, max: 15 }
                        }}
                        onUserMedia={() => console.log("✅ Webcam access granted")}
                        onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
                    />

                    {/* Face bounding boxes overlay */}
                    {Object.entries(faceTracks).map(([trackId, track]) => (
                        <div
                            key={trackId}
                            className="absolute border-2 border-green-400 rounded-lg"
                            style={{
                                left: `${track.bbox?.[0] || 0}px`,
                                top: `${track.bbox?.[1] || 0}px`,
                                width: `${track.bbox?.[2] || 0}px`,
                                height: `${track.bbox?.[3] || 0}px`,
                            }}
                        >
                            <div className="absolute -top-8 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
                                {track.personName} - {track.currentEmotion}
                                {track.gender !== 'unknown' && (
                                    <span className="ml-1 text-xs">({track.gender})</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-4 flex flex-col gap-3 w-full">
                    <button
                        onClick={downloadCSV}
                        disabled={emotionList.length === 0}
                        className={`px-4 py-2 rounded-lg transition ${emotionList.length === 0
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 text-white'
                            }`}
                    >
                        📥 Download CSV ({emotionList.length} entries)
                    </button>

                    <button
                        onClick={() => {
                            if (wsRef.current) {
                                wsRef.current.close();
                            }
                            setConnected(false);
                            setStatus("Reconnecting...");
                            setTimeout(connectWebSocket, 1000);
                        }}
                        className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
                    >
                        🔄 Reconnect
                    </button>
                </div>

                {/* Enhanced Face Tracking Info */}
                <div className="mt-4 w-full">
                    <h3 className="font-semibold text-gray-700 mb-2">
                        Active Faces: {Object.keys(faceTracks).length}
                    </h3>
                    <div className="text-xs text-gray-600 space-y-1 max-h-32 overflow-y-auto">
                        {Object.entries(faceTracks).length === 0 ? (
                            <p className="text-gray-400">No faces detected yet...</p>
                        ) : (
                            Object.entries(faceTracks).map(([trackId, track]) => (
                                <div key={trackId} className="flex justify-between items-center p-2 bg-gray-50 rounded border">
                                    <div className="flex items-center gap-2">
                                        <span className="font-medium">{track.personName}</span>
                                        <span className={`px-2 py-1 rounded text-xs capitalize ${track.currentEmotion === 'happy' ? 'bg-green-100 text-green-800' :
                                            track.currentEmotion === 'sad' ? 'bg-blue-100 text-blue-800' :
                                                track.currentEmotion === 'angry' ? 'bg-red-100 text-red-800' :
                                                    'bg-gray-100 text-gray-800'
                                            }`}>
                                            {track.currentEmotion}
                                        </span>
                                        {track.gender !== 'unknown' && (
                                            <span className={`text-xs px-2 py-1 rounded ${track.gender === 'man' ? 'bg-blue-100 text-blue-800' :
                                                'bg-pink-100 text-pink-800'
                                                }`}>
                                                {track.gender}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex gap-1 text-xs">
                                        <span className="bg-purple-100 px-1 rounded">#{track.detectionCount}</span>
                                        <span className="bg-orange-100 px-1 rounded">{Math.round(track.confidence * 100)}%</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Right Panel - Enhanced Emotion Log */}
            <div className="bg-white w-full md:w-96 h-[500px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
                <div className="flex justify-between items-center mb-3">
                    <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                        🧠 Emotion Log
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            DeepFace AI
                        </span>
                    </h3>
                    <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
                        {emotionList.length} entries
                    </span>
                </div>

                {emotionList.length === 0 ? (
                    <div className="text-center mt-20 text-gray-400">
                        <p className="text-xs mt-2">Powered by DeepFace AI:</p>
                        <ul className="text-xs text-left mt-1 space-y-1">
                            <li>• Face recognition & tracking</li>
                            <li>• Emotion & gender detection</li>
                            <li>• Multi-person analysis</li>
                        </ul>
                    </div>
                ) : (
                    <ul className="space-y-2">
                        {emotionList.map((e, index) => (
                            <li
                                key={index}
                                className={`flex justify-between items-start p-3 rounded-lg transition-all ${e.emotionChanged ? 'bg-green-50 border border-green-200' :
                                    e.isNewFace ? 'bg-yellow-50 border border-yellow-200' :
                                        'bg-gray-50 border border-gray-200'
                                    }`}
                            >
                                <div className="flex-1">
                                    <div className="flex justify-between items-start mb-1">
                                        <div className="flex items-center gap-2">
                                            <span className="font-medium capitalize text-gray-800">{e.emotion}</span>
                                            {e.gender && e.gender !== 'unknown' && (
                                                <span className={`text-xs px-2 py-1 rounded ${e.gender === 'man' ? 'bg-blue-100 text-blue-800' :
                                                    'bg-pink-100 text-pink-800'
                                                    }`}>
                                                    {e.gender}
                                                </span>
                                            )}
                                        </div>
                                        <div className="flex gap-1 text-xs">
                                            <span className="bg-blue-100 px-2 py-1 rounded">{e.confidence}</span>
                                            {e.genderConfidence && e.genderConfidence > 0.5 && (
                                                <span className="bg-purple-100 px-2 py-1 rounded">{e.genderConfidence}</span>
                                            )}
                                        </div>
                                    </div>
                                    <p className="text-xs text-gray-500 mb-2">{e.timestamp}</p>
                                    <div className="flex gap-2 flex-wrap">
                                        <span className="text-xs bg-gray-200 px-2 py-1 rounded">{e.personName}</span>
                                        <span className="text-xs bg-purple-200 px-2 py-1 rounded">Track: {e.track_count}</span>
                                        {e.similarity_score && (
                                            <span className="text-xs bg-blue-200 px-2 py-1 rounded">Similarity: {e.similarity_score}</span>
                                        )}
                                        {e.emotionChanged && (
                                            <span className="text-xs bg-green-200 px-2 py-1 rounded">Emotion Changed</span>
                                        )}
                                        {e.isNewFace && (
                                            <span className="text-xs bg-yellow-200 px-2 py-1 rounded">New Face</span>
                                        )}
                                    </div>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
// return (
//     <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
//         {/* Left Panel - Webcam & Controls */}
//         <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
//             <h2 className="text-2xl font-bold mb-2 text-gray-700">RetinaFace Emotion Tracker</h2>
//             <p className={`text-sm mb-3 font-medium ${getStatusColor()}`}>
//                 {getStatusEmoji()} {status}
//             </p>

//             {/* Webcam with face overlay */}
//             <div className="relative">
//                 <Webcam
//                     ref={webcamRef}
//                     width={480}
//                     height={360}
//                     className="rounded-lg border-2 border-gray-300"
//                     audio={false}
//                     screenshotFormat="image/jpeg"
//                     videoConstraints={{
//                         facingMode: "user",
//                         width: 480,
//                         height: 360,
//                         frameRate: { ideal: 10, max: 15 }
//                     }}
//                     onUserMedia={() => console.log("✅ Webcam access granted")}
//                     onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
//                 />

//                 {/* Face bounding boxes overlay */}
//                 {Object.values(faceTracks).map(track => (
//                     <div
//                         key={track.id}
//                         className="absolute border-2 border-green-400 rounded-lg"
//                         style={{
//                             left: `${track.bbox[0]}px`,
//                             top: `${track.bbox[1]}px`,
//                             width: `${track.bbox[2]}px`,
//                             height: `${track.bbox[3]}px`,
//                         }}
//                     >
//                         <div className="absolute -top-6 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
//                             User_{track.id} - {track.currentEmotion}
//                         </div>
//                     </div>
//                 ))}
//             </div>

//             <div className="mt-4 flex flex-col gap-3 w-full">
//                 <button
//                     onClick={downloadCSV}
//                     disabled={emotionList.length === 0}
//                     className={`px-4 py-2 rounded-lg transition ${emotionList.length === 0
//                         ? 'bg-gray-400 cursor-not-allowed'
//                         : 'bg-blue-600 hover:bg-blue-700 text-white'
//                         }`}
//                 >
//                     📥 Download CSV ({emotionList.length} entries)
//                 </button>

//                 <button
//                     onClick={() => {
//                         if (wsRef.current) {
//                             wsRef.current.close();
//                         }
//                         setConnected(false);
//                         setStatus("Reconnecting...");
//                         setTimeout(connectWebSocket, 1000);
//                     }}
//                     className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
//                 >
//                     🔄 Reconnect
//                 </button>
//             </div>

//             {/* Enhanced Face Tracking Info */}
//             <div className="mt-4 w-full">
//                 <h3 className="font-semibold text-gray-700 mb-2">
//                     Active Faces: {Object.keys(faceTracks).length}
//                 </h3>
//                 <div className="text-xs text-gray-600 space-y-1 max-h-32 overflow-y-auto">
//                     {Object.values(faceTracks).length === 0 ? (
//                         <p className="text-gray-400">No faces detected yet...</p>
//                     ) : (
//                         Object.values(faceTracks).map(track => (
//                             <div key={track.id} className="flex justify-between items-center p-2 bg-gray-50 rounded border">
//                                 <div className="flex items-center gap-2">
//                                     <span className="font-medium">User_{track.id}</span>
//                                     <span className={`px-2 py-1 rounded text-xs capitalize ${track.currentEmotion === 'happy' ? 'bg-green-100 text-green-800' :
//                                         track.currentEmotion === 'sad' ? 'bg-blue-100 text-blue-800' :
//                                             track.currentEmotion === 'angry' ? 'bg-red-100 text-red-800' :
//                                                 track.currentEmotion === 'surprise' ? 'bg-yellow-100 text-yellow-800' :
//                                                     'bg-gray-100 text-gray-800'
//                                         }`}>
//                                         {track.currentEmotion}
//                                     </span>
//                                 </div>
//                                 <div className="flex gap-1 text-xs">
//                                     <span className="bg-purple-100 px-1 rounded">#{track.detectionCount}</span>
//                                     <span className="bg-orange-100 px-1 rounded">{track.confidence}%</span>
//                                 </div>
//                             </div>
//                         ))
//                     )}
//                 </div>
//             </div>
//         </div>

//         {/* Right Panel - Enhanced Emotion Log */}
//         <div className="bg-white w-full md:w-96 h-[500px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
//             <div className="flex justify-between items-center mb-3">
//                 <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
//                     🧠 Emotion Log
//                     <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
//                         RetinaFace
//                     </span>
//                 </h3>
//                 <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
//                     {emotionList.length} entries
//                 </span>
//             </div>

//             {emotionList.length === 0 ? (
//                 <div className="text-center mt-20 text-gray-400">
//                     <p className="text-sm">No emotions detected yet...</p>
//                     <p className="text-xs mt-2">Powered by RetinaFace:</p>
//                     <ul className="text-xs text-left mt-1 space-y-1">
//                         <li>• State-of-the-art detection</li>
//                         <li>• Multiple face handling</li>
//                         <li>• High accuracy tracking</li>
//                     </ul>
//                 </div>
//             ) : (
//                 <ul className="space-y-2">
//                     {emotionList.map((e, index) => (
//                         <li
//                             key={index}
//                             className={`flex justify-between items-start p-3 rounded-lg transition-all ${e.emotionChanged ? 'bg-green-50 border border-green-200' :
//                                 e.isNewFace ? 'bg-yellow-50 border border-yellow-200' :
//                                     'bg-gray-50 border border-gray-200'
//                                 }`}
//                         >
//                             <div className="flex-1">
//                                 <div className="flex justify-between items-start mb-1">
//                                     <span className="font-medium capitalize text-gray-800">{e.emotion}</span>
//                                     <div className="flex gap-1 text-xs">
//                                         <span className="bg-blue-100 px-2 py-1 rounded">{e.confidence}</span>
//                                         <span className="bg-green-100 px-2 py-1 rounded">{e.detection_confidence}</span>
//                                     </div>
//                                 </div>
//                                 <p className="text-xs text-gray-500 mb-2">{e.timestamp}</p>
//                                 <div className="flex gap-2 flex-wrap">
//                                     <span className="text-xs bg-gray-200 px-2 py-1 rounded">ID: {e.faceId}</span>
//                                     <span className="text-xs bg-purple-200 px-2 py-1 rounded">Track: {e.track_count}</span>
//                                     {e.emotionChanged && (
//                                         <span className="text-xs bg-green-200 px-2 py-1 rounded">Emotion Changed</span>
//                                     )}
//                                     {e.isNewFace && (
//                                         <span className="text-xs bg-yellow-200 px-2 py-1 rounded">New Face</span>
//                                     )}
//                                 </div>
//                             </div>
//                         </li>
//                     ))}
//                 </ul>
//             )}
//         </div>
//     </div>
//     // <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
//     //     {/* Left Panel - Webcam & Controls */}
//     //     <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
//     //         <h2 className="text-2xl font-bold mb-2 text-gray-700">Python Emotion Tracker</h2>
//     //         <p className={`text-sm mb-3 font-medium ${getStatusColor()}`}>
//     //             {getStatusEmoji()} {status}
//     //         </p>

//     //         <Webcam
//     //             ref={webcamRef}
//     //             width={480}
//     //             height={360}
//     //             className="rounded-lg border-2 border-gray-300"
//     //             audio={false}
//     //             screenshotFormat="image/jpeg"
//     //             videoConstraints={{
//     //                 facingMode: "user",
//     //                 width: 480,
//     //                 height: 360,
//     //                 frameRate: { ideal: 10, max: 15 }
//     //             }}
//     //             onUserMedia={() => console.log("✅ Webcam access granted")}
//     //             onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
//     //         />

//     //         <div className="mt-4 flex flex-col gap-3 w-full">
//     //             <button
//     //                 onClick={downloadCSV}
//     //                 disabled={emotionList.length === 0}
//     //                 className={`px-4 py-2 rounded-lg transition ${emotionList.length === 0
//     //                     ? 'bg-gray-400 cursor-not-allowed'
//     //                     : 'bg-blue-600 hover:bg-blue-700 text-white'
//     //                     }`}
//     //             >
//     //                 📥 Download CSV ({emotionList.length} entries)
//     //             </button>

//     //             <button
//     //                 onClick={() => {
//     //                     if (wsRef.current) {
//     //                         wsRef.current.close();
//     //                     }
//     //                     setConnected(false);
//     //                     setStatus("Reconnecting...");
//     //                     setTimeout(connectWebSocket, 1000);
//     //                 }}
//     //                 className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
//     //             >
//     //                 🔄 Reconnect
//     //             </button>
//     //         </div>

//     //         {/* Face Tracking Info */}
//     //         <div className="mt-4 w-full">
//     //             <h3 className="font-semibold text-gray-700 mb-2">
//     //                 Active Faces: {Object.keys(faceTracks).length}
//     //             </h3>
//     //             <div className="text-xs text-gray-600 space-y-1 max-h-32 overflow-y-auto">
//     //                 {Object.values(faceTracks).length === 0 ? (
//     //                     <p className="text-gray-400">No faces detected yet...</p>
//     //                 ) : (
//     //                     Object.values(faceTracks).map(track => (
//     //                         <div key={track.id} className="flex justify-between items-center p-1 bg-gray-50 rounded">
//     //                             <span className="font-medium">User_{track.id}</span>
//     //                             <span className="capitalize px-2 py-1 bg-blue-100 rounded">{track.currentEmotion}</span>
//     //                             <span className="text-gray-500">({track.detectionCount})</span>
//     //                         </div>
//     //                     ))
//     //                 )}
//     //             </div>
//     //         </div>
//     //     </div>

//     //     {/* Right Panel - Enhanced Emotion Log */}
//     //     <div className="bg-white w-full md:w-96 h-[500px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
//     //         <div className="flex justify-between items-center mb-3">
//     //             <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
//     //                 🧠 Emotion Log
//     //             </h3>
//     //             <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
//     //                 {emotionList.length} entries
//     //             </span>
//     //         </div>

//     //         {emotionList.length === 0 ? (
//     //             <div className="text-center mt-20 text-gray-400">
//     //                 <p className="text-sm">No emotions detected yet...</p>
//     //                 <p className="text-xs mt-2">Make sure:</p>
//     //                 <ul className="text-xs text-left mt-1 space-y-1">
//     //                     <li>• Python backend is running</li>
//     //                     <li>• Webcam is enabled</li>
//     //                     <li>• Good lighting on your face</li>
//     //                 </ul>
//     //             </div>
//     //         ) : (
//     //             <ul className="space-y-2">
//     //                 {emotionList.map((e, index) => (
//     //                     <li
//     //                         key={index}
//     //                         className={`flex justify-between items-start p-3 rounded-lg transition-all ${e.emotionChanged
//     //                             ? 'bg-green-50 border border-green-200'
//     //                             : e.isNewFace
//     //                                 ? 'bg-yellow-50 border border-yellow-200'
//     //                                 : 'bg-gray-50 border border-gray-200'
//     //                             }`}
//     //                     >
//     //                         <div className="flex-1">
//     //                             <div className="flex justify-between items-start mb-1">
//     //                                 <span className="font-medium capitalize text-gray-800">{e.emotion}</span>
//     //                                 <span className="text-sm font-semibold text-blue-600">{e.confidence}</span>
//     //                             </div>
//     //                             <p className="text-xs text-gray-500 mb-2">{e.timestamp}</p>
//     //                             <div className="flex gap-2 flex-wrap">
//     //                                 <span className="text-xs bg-gray-200 px-2 py-1 rounded">{e.faceId}</span>
//     //                                 {e.emotionChanged && (
//     //                                     <span className="text-xs bg-green-200 px-2 py-1 rounded">Emotion Changed</span>
//     //                                 )}
//     //                                 {e.isNewFace && (
//     //                                     <span className="text-xs bg-yellow-200 px-2 py-1 rounded">New Face</span>
//     //                                 )}
//     //                             </div>
//     //                         </div>
//     //                     </li>
//     //                 ))}
//     //             </ul>
//     //         )}
//     //     </div>
//     // </div>
// );
