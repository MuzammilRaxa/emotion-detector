"use client";
import { getWebSocketUrl } from "@/lib/websockets";
import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

export default function PremiumEmotionTracker() {
    const webcamRef = useRef(null);
    const [status, setStatus] = useState("Idle");
    const [trackingList, setTrackingList] = useState([]);
    const [connected, setConnected] = useState(false);
    const [faceTracks, setFaceTracks] = useState({});
    const [emotionAnalytics, setEmotionAnalytics] = useState({});
    const wsRef = useRef(null);
    const animationRef = useRef(null);
    const frameCounterRef = useRef(0);
    const MAX_ENTRIES = 500;

    // Premium emotion categories with enhanced colors
    const getEmotionColor = (emotion) => {
        const emotionColors = {
            // Core emotions with premium colors
            happy: "bg-yellow-100 text-yellow-800 border-yellow-300",
            smiling: "bg-yellow-50 text-yellow-700 border-yellow-200",
            laughing: "bg-yellow-200 text-yellow-900 border-yellow-400",
            sad: "bg-blue-100 text-blue-800 border-blue-300",
            crying: "bg-blue-200 text-blue-900 border-blue-400",
            angry: "bg-red-100 text-red-800 border-red-300",
            furious: "bg-red-200 text-red-900 border-red-400",
            surprise: "bg-purple-100 text-purple-800 border-purple-300",
            surprised: "bg-purple-50 text-purple-700 border-purple-200",
            fear: "bg-indigo-100 text-indigo-800 border-indigo-300",
            scared: "bg-indigo-200 text-indigo-900 border-indigo-400",
            disgust: "bg-green-100 text-green-800 border-green-300",
            neutral: "bg-gray-100 text-gray-800 border-gray-300",
            calm: "bg-gray-50 text-gray-700 border-gray-200"
        };
        return emotionColors[emotion] || "bg-gray-100 text-gray-800 border-gray-300";
    };

    const getEmotionEmoji = (emotion) => {
        const emotionEmojis = {
            happy: "😊",
            smiling: "😄",
            laughing: "😂",
            sad: "😢",
            crying: "😭",
            angry: "😠",
            furious: "😡",
            surprise: "😲",
            surprised: "😮",
            fear: "😨",
            scared: "😰",
            disgust: "🤢",
            neutral: "😐",
            calm: "😌"
        };
        return emotionEmojis[emotion] || "❓";
    };

    const getConfidenceLevel = (confidence) => {
        if (confidence >= 0.90) return { text: "Premium", color: "bg-green-100 text-green-800", emoji: "🎯" };
        if (confidence >= 0.85) return { text: "High", color: "bg-blue-100 text-blue-800", emoji: "✅" };
        if (confidence >= 0.70) return { text: "Good", color: "bg-yellow-100 text-yellow-800", emoji: "⚠️" };
        return { text: "Low", color: "bg-red-100 text-red-800", emoji: "❌" };
    };

    const formatDuration = (seconds) => {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        }
    };

    const getStabilityLevel = (stability) => {
        if (stability >= 0.8) return { text: "Very Stable", color: "bg-green-100 text-green-800" };
        if (stability >= 0.6) return { text: "Stable", color: "bg-blue-100 text-blue-800" };
        if (stability >= 0.4) return { text: "Moderate", color: "bg-yellow-100 text-yellow-800" };
        return { text: "Unstable", color: "bg-red-100 text-red-800" };
    };

    // Track seen faces for proper new face detection
    const [seenFaceIds, setSeenFaceIds] = useState(new Set());

    const updateSeenFaces = (faceId, personName) => {
        setSeenFaceIds(prev => {
            const newSet = new Set(prev);
            newSet.add(`${faceId}_${personName}`);
            return newSet;
        });
    };

    const isFaceTrulyNew = (faceId, personName) => {
        return !seenFaceIds.has(`${faceId}_${personName}`);
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
    setStatus("Connecting to premium emotion server...");

    try {
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL;
        console.log('Connecting to:', wsUrl);

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        let heartbeatInterval;
        let reconnectTimeout;

        ws.onopen = () => {
            console.log("✅ Premium WebSocket connected successfully");
            setConnected(true);
            setStatus("Connected! Starting premium emotion tracking...");
            
            // Start heartbeat
            heartbeatInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 15000); // Send ping every 15 seconds
            
            startFrameCapture();
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'connection_established') {
                    console.log('🔗 Server connection confirmed:', data.message);
                    return;
                }

                if (data.type === 'pong') {
                    console.log('💓 Heartbeat received');
                    return;
                }

                if (data.type === 'premium_emotion_results') {
                    // Your existing emotion processing code...
                    const newFaceTracks = {};
                    const newEmotionAnalytics = {};

                    data.detections.forEach(detection => {
                        const faceId = detection.face_id;
                        const personName = detection.person_name || `Person_${detection.face_id}`;
                        
                        // Update face tracks
                        newFaceTracks[faceId] = {
                            id: faceId,
                            personName: personName,
                            bbox: detection.bbox || [0, 0, 0, 0],
                            detectionCount: detection.track_count || 1,
                            currentEmotion: detection.current_emotion || 'neutral',
                            emotionConfidence: detection.emotion_confidence || 0,
                            emotionCategory: detection.emotion_category || 'neutral',
                            isHighConfidence: detection.is_high_confidence || false,
                            emotionChanged: detection.emotion_changed || false,
                            lastSeen: Date.now(),
                            confirmed: detection.confirmed || false
                        };

                        // Update emotion analytics
                        newEmotionAnalytics[faceId] = {
                            stability: detection.emotion_stability || 0,
                            currentDuration: detection.current_emotion_duration || 0,
                            totalDuration: detection.total_emotion_duration || 0,
                            dominantEmotion: detection.dominant_emotion || 'unknown',
                            averageConfidence: detection.average_confidence || 0,
                            totalTransitions: detection.total_transitions || 0,
                            highConfidenceRatio: detection.high_confidence_ratio || 0,
                            recentTrend: detection.recent_emotion_trend || [],
                            context: detection.context || {}
                        };

                        updateSeenFaces(faceId, personName);
                    });

                    setFaceTracks(newFaceTracks);
                    setEmotionAnalytics(newEmotionAnalytics);

                    // Update tracking list
                    if (data.detections.length > 0) {
                        setTrackingList(prev => {
                            const newEntries = data.detections.map(detection => {
                                const faceId = detection.face_id;
                                const personName = detection.person_name || `Person_${detection.face_id}`;
                                const isNewFace = isFaceTrulyNew(faceId, personName);
                                
                                return {
                                    personName: personName,
                                    faceId: faceId,
                                    bbox: detection.bbox || [],
                                    timestamp: detection.timestamp || new Date().toISOString(),
                                    track_count: detection.track_count || 1,
                                    current_emotion: detection.current_emotion || 'neutral',
                                    emotion_confidence: detection.emotion_confidence || 0,
                                    emotion_category: detection.emotion_category || 'neutral',
                                    is_high_confidence: detection.is_high_confidence || false,
                                    emotion_changed: detection.emotion_changed || false,
                                    emotion_stability: detection.emotion_stability || 0,
                                    current_emotion_duration: detection.current_emotion_duration || 0,
                                    total_emotion_duration: detection.total_emotion_duration || 0,
                                    dominant_emotion: detection.dominant_emotion || 'unknown',
                                    total_transitions: detection.total_transitions || 0,
                                    high_confidence_ratio: detection.high_confidence_ratio || 0,
                                    recent_trend: detection.recent_emotion_trend || [],
                                    context: detection.context || {},
                                    confirmed: detection.confirmed || false,
                                    isNewFace: isNewFace,
                                    emotion_record: detection.emotion_record || null
                                };
                            });

                            const updatedList = [...newEntries, ...prev].slice(0, MAX_ENTRIES);
                            return updatedList;
                        });
                    }

                    console.log(`🎯 Premium Update: ${data.high_confidence_tracks || 0} high-confidence emotions`);
                }

                if (data.type === 'error') {
                    console.error('Server error:', data.message);
                }

            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        ws.onclose = (event) => {
            console.log("📞 WebSocket disconnected:", event.code, event.reason);
            setConnected(false);
            
            // Clear intervals
            if (heartbeatInterval) clearInterval(heartbeatInterval);
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
            
            // Don't show error for normal closure
            if (event.code !== 1000) {
                setStatus(`Disconnected (${event.code}). Retrying in 5s...`);
                
                // Reconnect after delay
                reconnectTimeout = setTimeout(() => {
                    if (!connected) {
                        console.log('🔄 Attempting to reconnect...');
                        connectWebSocket();
                    }
                }, 5000);
            } else {
                setStatus("Disconnected");
            }
        };

        ws.onerror = (error) => {
            console.log("❌ WebSocket error:", error);
            setStatus("Connection error - Check if server is running");
            setConnected(false);
        };

    } catch (error) {
        console.log("❌ Failed to create WebSocket:", error);
        setStatus("Failed to connect - Retrying...");
        setTimeout(connectWebSocket, 5000);
    }
};

    const startFrameCapture = () => {
        let lastSent = 0;
        const FPS = 15; // Increased FPS for smoother tracking

        const captureFrame = () => {
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

            if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const frameData = canvas.toDataURL('image/jpeg', 0.8); // Higher quality for premium
                    frameCounterRef.current++;

                    if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({
                            type: 'frame',
                            frame: frameData,
                            timestamp: now,
                            frameId: frameCounterRef.current,
                            quality: 'premium'
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

    const generatePremiumCSV = () => {
        const headers = [
            "Timestamp", "Person Name", "Face ID", "Current Emotion", "Emotion Category",
            "Emotion Confidence", "Confidence Level", "High Confidence", "Emotion Stability",
            "Current Duration", "Total Duration", "Dominant Emotion", "Total Transitions",
            "High Confidence Ratio", "Track Count", "Confirmed"
        ];

        const rows = trackingList.map(track => [
            track.timestamp,
            track.personName,
            track.faceId,
            track.current_emotion,
            track.emotion_category,
            (track.emotion_confidence * 100).toFixed(1) + '%',
            getConfidenceLevel(track.emotion_confidence).text,
            track.is_high_confidence ? 'YES' : 'NO',
            (track.emotion_stability * 100).toFixed(1) + '%',
            formatDuration(track.current_emotion_duration),
            formatDuration(track.total_emotion_duration),
            track.dominant_emotion,
            track.total_transitions,
            (track.high_confidence_ratio * 100).toFixed(1) + '%',
            track.track_count,
            track.confirmed ? 'YES' : 'NO'
        ]);

        const csvContent = [headers, ...rows].map(r => r.join(",")).join("\n");
        return csvContent;
    };

    const downloadCSV = () => {
        if (trackingList.length === 0) {
            alert("No tracking data to download yet!");
            return;
        }

        const csv = generatePremiumCSV();
        const blob = new Blob([csv], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `premium_emotion_tracking_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    };

    const clearAllRecords = () => {
        if (trackingList.length === 0) {
            alert("No records to clear!");
            return;
        }

        const confirmClear = confirm(
            `Are you sure you want to clear all ${trackingList.length} premium emotion records?`
        );

        if (confirmClear) {
            setTrackingList([]);
            setFaceTracks({});
            setEmotionAnalytics({});
            setSeenFaceIds(new Set());
            alert("All premium records cleared!");
        }
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
        <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 min-h-screen">
            {/* Left Panel - Webcam & Premium Controls */}
            <div className="flex flex-col items-center bg-white rounded-2xl shadow-xl p-6 w-full md:w-auto">
                <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                    Premium Emotion AI
                </h2>
                <p className={`text-sm mb-4 font-medium ${getStatusColor()} flex items-center gap-2`}>
                    <span className="text-lg">{getStatusEmoji()}</span>
                    {status}
                </p>

                {/* Webcam with premium overlay */}
                <div className="relative">
                    <Webcam
                        ref={webcamRef}
                        width={540}
                        height={405}
                        className="rounded-xl border-4 border-purple-200 shadow-lg"
                        audio={false}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{
                            facingMode: "user",
                            width: 540,
                            height: 405,
                            frameRate: { ideal: 15, max: 20 }
                        }}
                        onUserMedia={() => console.log("✅ HD Webcam access granted")}
                        onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
                    />

                    {/* Premium face bounding boxes */}
                    {Object.entries(faceTracks).map(([trackId, track]) => {
                        const analytics = emotionAnalytics[trackId] || {};
                        const confidenceInfo = getConfidenceLevel(track.emotionConfidence);
                        const stabilityInfo = getStabilityLevel(analytics.stability || 0);
                        
                        return (
                            <div
                                key={trackId}
                                className="absolute border-3 border-green-400 rounded-lg shadow-2xl"
                                style={{
                                    left: `${track.bbox?.[0] || 0}px`,
                                    top: `${track.bbox?.[1] || 0}px`,
                                    width: `${track.bbox?.[2] || 0}px`,
                                    height: `${track.bbox?.[3] || 0}px`,
                                }}
                            >
                                <div className="absolute -top-10 left-0 bg-gradient-to-r from-green-500 to-blue-500 text-white text-sm px-3 py-2 rounded-lg shadow-lg">
                                    <div className="flex items-center gap-2">
                                        <span className="text-lg">{getEmotionEmoji(track.currentEmotion)}</span>
                                        <span className="font-bold">{track.personName}</span>
                                        {track.emotionChanged && (
                                            <span className="ml-1 animate-pulse text-yellow-300">🔄</span>
                                        )}
                                        {track.isHighConfidence && (
                                            <span className="ml-1 text-yellow-300">🎯</span>
                                        )}
                                    </div>
                                    <div className="text-xs opacity-90">
                                        {confidenceInfo.emoji} {(track.emotionConfidence * 100).toFixed(0)}% • 
                                        {stabilityInfo.text} • {formatDuration(analytics.currentDuration || 0)}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Premium Control Buttons */}
                <div className="mt-6 flex flex-col gap-3 w-full">
                    <button
                        onClick={downloadCSV}
                        disabled={trackingList.length === 0}
                        className={`px-6 py-3 rounded-xl transition-all flex items-center justify-center gap-3 font-semibold ${
                            trackingList.length === 0
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl'
                        }`}
                    >
                        <span className="text-xl">📊</span>
                        <span>Export Premium Analytics ({trackingList.length})</span>
                    </button>

                    <button
                        onClick={clearAllRecords}
                        disabled={trackingList.length === 0}
                        className={`px-6 py-3 rounded-xl transition-all flex items-center justify-center gap-3 font-semibold ${
                            trackingList.length === 0
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white shadow-lg hover:shadow-xl'
                        }`}
                    >
                        <span className="text-xl">🗑️</span>
                        <span>Clear All Records</span>
                    </button>

                    <button
                        onClick={() => {
                            if (wsRef.current) {
                                wsRef.current.close();
                            }
                            setConnected(false);
                            setStatus("Reconnecting to premium server...");
                            setTimeout(connectWebSocket, 1000);
                        }}
                        className="px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl hover:from-gray-700 hover:to-gray-800 transition-all flex items-center justify-center gap-3 font-semibold shadow-lg hover:shadow-xl"
                    >
                        <span className="text-xl">🔄</span>
                        <span>Reconnect</span>
                    </button>
                </div>

                {/* Premium Analytics Summary */}
                <div className="mt-6 w-full">
                    <h3 className="font-bold text-gray-800 mb-3 text-lg">Live Analytics</h3>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                            <div className="text-blue-800 font-semibold">High Confidence</div>
                            <div className="text-2xl font-bold text-blue-600">
                                {Object.values(faceTracks).filter(t => t.isHighConfidence).length}
                            </div>
                        </div>
                        <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                            <div className="text-green-800 font-semibold">Total Tracks</div>
                            <div className="text-2xl font-bold text-green-600">
                                {Object.keys(faceTracks).length}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Right Panel - Premium Emotion Analytics */}
            <div className="bg-white w-full md:w-[500px] h-[600px] rounded-2xl shadow-xl p-6 overflow-y-auto border border-purple-200">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center gap-3">
                        <span className="text-2xl">🎭</span>
                        Premium Emotion Analytics
                        <span className="text-xs bg-gradient-to-r from-purple-100 to-blue-100 text-purple-800 px-3 py-1 rounded-full">
                            90%+ Accuracy
                        </span>
                    </h3>
                    <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full font-semibold">
                        {trackingList.length} entries
                    </span>
                </div>

                {trackingList.length === 0 ? (
                    <div className="text-center mt-20 text-gray-500">
                        <div className="text-6xl mb-6">😊</div>
                        <p className="text-lg font-semibold mb-2">Premium Emotion Tracking Ready</p>
                        <p className="text-sm mb-6">High-confidence emotion detection will appear here</p>
                        <div className="text-left text-sm space-y-3 max-w-xs mx-auto bg-blue-50 p-4 rounded-xl">
                            <p className="font-bold text-blue-800">Premium Features:</p>
                            <ul className="space-y-2">
                                <li className="flex items-center gap-2">🎯 90%+ Confidence Tracking</li>
                                <li className="flex items-center gap-2">📊 Emotion Stability Analysis</li>
                                <li className="flex items-center gap-2">⏱️ Duration & Frequency</li>
                                <li className="flex items-center gap-2">🔄 Transition Tracking</li>
                                <li className="flex items-center gap-2">💡 HD Camera Optimized</li>
                            </ul>
                        </div>
                    </div>
                ) : (
                    <>
                        <ul className="space-y-4">
                            {trackingList.map((track, index) => {
                                const confidenceInfo = getConfidenceLevel(track.emotion_confidence);
                                const stabilityInfo = getStabilityLevel(track.emotion_stability);
                                
                                return (
                                    <li
                                        key={index}
                                        className={`flex justify-between items-start p-4 rounded-xl transition-all border-2 ${
                                            track.is_high_confidence 
                                                ? 'bg-gradient-to-r from-green-50 to-blue-50 border-green-200 shadow-md' 
                                                : track.emotion_changed 
                                                    ? 'bg-blue-50 border-blue-200' 
                                                    : 'bg-gray-50 border-gray-200'
                                        }`}
                                    >
                                        <div className="flex-1">
                                            <div className="flex justify-between items-start mb-3">
                                                <div className="flex items-center gap-3">
                                                    <span className="font-bold text-gray-900">{track.personName}</span>
                                                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getEmotionColor(track.current_emotion)}`}>
                                                        <span className="text-lg mr-1">{getEmotionEmoji(track.current_emotion)}</span>
                                                        {track.current_emotion}
                                                    </span>
                                                </div>
                                                <div className="flex gap-2">
                                                    <span className={`px-2 py-1 rounded text-xs ${confidenceInfo.color}`}>
                                                        {confidenceInfo.emoji} {(track.emotion_confidence * 100).toFixed(0)}%
                                                    </span>
                                                    {track.confirmed && (
                                                        <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">✓</span>
                                                    )}
                                                </div>
                                            </div>
                                            
                                            <p className="text-xs text-gray-500 mb-3">{track.timestamp}</p>
                                            
                                            {/* Premium Analytics Row */}
                                            <div className="grid grid-cols-2 gap-2 mb-3">
                                                <div className={`px-2 py-1 rounded text-xs ${stabilityInfo.color}`}>
                                                    Stability: {stabilityInfo.text}
                                                </div>
                                                <div className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs">
                                                    ⏱️ {formatDuration(track.current_emotion_duration)}
                                                </div>
                                                <div className="bg-orange-100 text-orange-800 px-2 py-1 rounded text-xs">
                                                    🔄 {track.total_transitions} transitions
                                                </div>
                                                <div className="bg-cyan-100 text-cyan-800 px-2 py-1 rounded text-xs">
                                                    📈 {(track.high_confidence_ratio * 100).toFixed(0)}% reliable
                                                </div>
                                            </div>

                                            {/* Context Information */}
                                            {track.context && (track.context.estimated_age || track.context.gender) && (
                                                <div className="bg-gray-100 p-2 rounded-lg text-xs">
                                                    <div className="font-semibold text-gray-700">Context:</div>
                                                    <div className="flex gap-4 mt-1">
                                                        {track.context.estimated_age && (
                                                            <span>Age: ~{track.context.estimated_age}</span>
                                                        )}
                                                        {track.context.gender && (
                                                            <span>Gender: {track.context.gender}</span>
                                                        )}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Recent Trend */}
                                            {track.recent_trend && track.recent_trend.length > 0 && (
                                                <div className="mt-2">
                                                    <div className="text-xs text-gray-600 font-semibold">Recent Trend:</div>
                                                    <div className="flex gap-1 mt-1">
                                                        {track.recent_trend.slice(-5).map((emotion, idx) => (
                                                            <span key={idx} className="text-xs bg-white px-2 py-1 rounded border">
                                                                {getEmotionEmoji(emotion)}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </li>
                                );
                            })}
                        </ul>
                        
                        <div className="mt-6 pt-4 border-t border-gray-200">
                            <button
                                onClick={clearAllRecords}
                                className="w-full px-6 py-3 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white rounded-xl transition-all flex items-center justify-center gap-3 font-semibold shadow-lg"
                            >
                                <span className="text-xl">🗑️</span>
                                <span>Clear All {trackingList.length} Records</span>
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
// "use client";
// import { useEffect, useRef, useState } from "react";
// import Webcam from "react-webcam";

// export default function PremiumEmotionTracker() {
//     const webcamRef = useRef(null);
//     const [status, setStatus] = useState("Idle");
//     const [trackingList, setTrackingList] = useState([]);
//     const [connected, setConnected] = useState(false);
//     const [faceTracks, setFaceTracks] = useState({});
//     const [emotionAnalytics, setEmotionAnalytics] = useState({});
//     const wsRef = useRef(null);
//     const animationRef = useRef(null);
//     const frameCounterRef = useRef(0);
//     const MAX_ENTRIES = 500;

//     // Premium emotion categories with enhanced colors
//     const getEmotionColor = (emotion) => {
//         const emotionColors = {
//             // Core emotions with premium colors
//             happy: "bg-yellow-100 text-yellow-800 border-yellow-300",
//             smiling: "bg-yellow-50 text-yellow-700 border-yellow-200",
//             laughing: "bg-yellow-200 text-yellow-900 border-yellow-400",
//             sad: "bg-blue-100 text-blue-800 border-blue-300",
//             crying: "bg-blue-200 text-blue-900 border-blue-400",
//             angry: "bg-red-100 text-red-800 border-red-300",
//             furious: "bg-red-200 text-red-900 border-red-400",
//             surprise: "bg-purple-100 text-purple-800 border-purple-300",
//             surprised: "bg-purple-50 text-purple-700 border-purple-200",
//             fear: "bg-indigo-100 text-indigo-800 border-indigo-300",
//             scared: "bg-indigo-200 text-indigo-900 border-indigo-400",
//             disgust: "bg-green-100 text-green-800 border-green-300",
//             neutral: "bg-gray-100 text-gray-800 border-gray-300",
//             calm: "bg-gray-50 text-gray-700 border-gray-200"
//         };
//         return emotionColors[emotion] || "bg-gray-100 text-gray-800 border-gray-300";
//     };

//     const getEmotionEmoji = (emotion) => {
//         const emotionEmojis = {
//             happy: "😊",
//             smiling: "😄",
//             laughing: "😂",
//             sad: "😢",
//             crying: "😭",
//             angry: "😠",
//             furious: "😡",
//             surprise: "😲",
//             surprised: "😮",
//             fear: "😨",
//             scared: "😰",
//             disgust: "🤢",
//             neutral: "😐",
//             calm: "😌"
//         };
//         return emotionEmojis[emotion] || "❓";
//     };

//     const getConfidenceLevel = (confidence) => {
//         if (confidence >= 0.90) return { text: "Premium", color: "bg-green-100 text-green-800", emoji: "🎯" };
//         if (confidence >= 0.85) return { text: "High", color: "bg-blue-100 text-blue-800", emoji: "✅" };
//         if (confidence >= 0.70) return { text: "Good", color: "bg-yellow-100 text-yellow-800", emoji: "⚠️" };
//         return { text: "Low", color: "bg-red-100 text-red-800", emoji: "❌" };
//     };

//     const formatDuration = (seconds) => {
//         if (seconds < 60) {
//             return `${Math.round(seconds)}s`;
//         } else {
//             const minutes = Math.floor(seconds / 60);
//             const remainingSeconds = Math.round(seconds % 60);
//             return `${minutes}m ${remainingSeconds}s`;
//         }
//     };

//     const getStabilityLevel = (stability) => {
//         if (stability >= 0.8) return { text: "Very Stable", color: "bg-green-100 text-green-800" };
//         if (stability >= 0.6) return { text: "Stable", color: "bg-blue-100 text-blue-800" };
//         if (stability >= 0.4) return { text: "Moderate", color: "bg-yellow-100 text-yellow-800" };
//         return { text: "Unstable", color: "bg-red-100 text-red-800" };
//     };

//     // Track seen faces for proper new face detection
//     const [seenFaceIds, setSeenFaceIds] = useState(new Set());

//     const updateSeenFaces = (faceId, personName) => {
//         setSeenFaceIds(prev => {
//             const newSet = new Set(prev);
//             newSet.add(`${faceId}_${personName}`);
//             return newSet;
//         });
//     };

//     const isFaceTrulyNew = (faceId, personName) => {
//         return !seenFaceIds.has(`${faceId}_${personName}`);
//     };

//     useEffect(() => {
//         connectWebSocket();

//         return () => {
//             if (wsRef.current) {
//                 wsRef.current.close();
//             }
//             if (animationRef.current) {
//                 cancelAnimationFrame(animationRef.current);
//             }
//         };
//     }, []);

//     const connectWebSocket = () => {
//         setStatus("Connecting to premium emotion server...");

//         try {
//             const ws = new WebSocket("ws://localhost:8765");
//             wsRef.current = ws;

//             ws.onopen = () => {
//                 console.log("✅ Premium WebSocket connected successfully");
//                 setConnected(true);
//                 setStatus("Connected! Starting premium emotion tracking...");
//                 startFrameCapture();
//             };

//             ws.onmessage = (event) => {
//                 try {
//                     const data = JSON.parse(event.data);

//                     if (data.type === 'premium_emotion_results') {
//                         const newFaceTracks = {};
//                         const newEmotionAnalytics = {};

//                         data.detections.forEach(detection => {
//                             const faceId = detection.face_id;
//                             const personName = detection.person_name || `Person_${detection.face_id}`;
                            
//                             // Update face tracks
//                             newFaceTracks[faceId] = {
//                                 id: faceId,
//                                 personName: personName,
//                                 bbox: detection.bbox || [0, 0, 0, 0],
//                                 detectionCount: detection.track_count || 1,
//                                 currentEmotion: detection.current_emotion || 'neutral',
//                                 emotionConfidence: detection.emotion_confidence || 0,
//                                 emotionCategory: detection.emotion_category || 'neutral',
//                                 isHighConfidence: detection.is_high_confidence || false,
//                                 emotionChanged: detection.emotion_changed || false,
//                                 lastSeen: Date.now(),
//                                 confirmed: detection.confirmed || false
//                             };

//                             // Update emotion analytics
//                             newEmotionAnalytics[faceId] = {
//                                 stability: detection.emotion_stability || 0,
//                                 currentDuration: detection.current_emotion_duration || 0,
//                                 totalDuration: detection.total_emotion_duration || 0,
//                                 dominantEmotion: detection.dominant_emotion || 'unknown',
//                                 averageConfidence: detection.average_confidence || 0,
//                                 totalTransitions: detection.total_transitions || 0,
//                                 highConfidenceRatio: detection.high_confidence_ratio || 0,
//                                 recentTrend: detection.recent_emotion_trend || [],
//                                 context: detection.context || {}
//                             };

//                             // Update seen faces
//                             updateSeenFaces(faceId, personName);
//                         });

//                         setFaceTracks(newFaceTracks);
//                         setEmotionAnalytics(newEmotionAnalytics);

//                         // Update tracking list with premium data
//                         if (data.detections.length > 0) {
//                             setTrackingList(prev => {
//                                 const newEntries = data.detections.map(detection => {
//                                     const faceId = detection.face_id;
//                                     const personName = detection.person_name || `Person_${detection.face_id}`;
                                    
//                                     const isNewFace = isFaceTrulyNew(faceId, personName);
                                    
//                                     return {
//                                         personName: personName,
//                                         faceId: faceId,
//                                         bbox: detection.bbox || [],
//                                         timestamp: detection.timestamp || new Date().toISOString(),
//                                         track_count: detection.track_count || 1,
                                        
//                                         // Premium emotion data
//                                         current_emotion: detection.current_emotion || 'neutral',
//                                         emotion_confidence: detection.emotion_confidence || 0,
//                                         emotion_category: detection.emotion_category || 'neutral',
//                                         is_high_confidence: detection.is_high_confidence || false,
//                                         emotion_changed: detection.emotion_changed || false,
                                        
//                                         // Analytics
//                                         emotion_stability: detection.emotion_stability || 0,
//                                         current_emotion_duration: detection.current_emotion_duration || 0,
//                                         total_emotion_duration: detection.total_emotion_duration || 0,
//                                         dominant_emotion: detection.dominant_emotion || 'unknown',
//                                         total_transitions: detection.total_transitions || 0,
//                                         high_confidence_ratio: detection.high_confidence_ratio || 0,
//                                         recent_trend: detection.recent_emotion_trend || [],
                                        
//                                         // Context
//                                         context: detection.context || {},
                                        
//                                         // Status
//                                         confirmed: detection.confirmed || false,
//                                         isNewFace: isNewFace,
//                                         emotion_record: detection.emotion_record || null
//                                     };
//                                 });

//                                 const updatedList = [...newEntries, ...prev].slice(0, MAX_ENTRIES);
//                                 return updatedList;
//                             });
//                         }

//                         console.log(`🎯 Premium Update: ${data.high_confidence_tracks || 0} high-confidence emotions`);
//                     }
//                 } catch (error) {
//                     console.error('Error processing WebSocket message:', error);
//                 }
//             };

//             ws.onclose = (event) => {
//                 console.log("📞 WebSocket disconnected:", event.code, event.reason);
//                 setConnected(false);
//                 setStatus(`Disconnected (${event.code}). Retrying in 3s...`);

//                 setTimeout(() => {
//                     if (!connected) {
//                         connectWebSocket();
//                     }
//                 }, 3000);
//             };

//             ws.onerror = (error) => {
//                 console.log("❌ WebSocket error:", error);
//                 setStatus("Connection error - Check if Python server is running");
//                 setConnected(false);
//             };
//         } catch (error) {
//             console.log("❌ Failed to create WebSocket:", error);
//             setStatus("Failed to connect - Retrying...");
//             setTimeout(connectWebSocket, 3000);
//         }
//     };

//     const startFrameCapture = () => {
//         let lastSent = 0;
//         const FPS = 15; // Increased FPS for smoother tracking

//         const captureFrame = () => {
//             if (!connected || !webcamRef.current?.video || wsRef.current?.readyState !== WebSocket.OPEN) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const now = Date.now();
//             if (now - lastSent < 1000 / FPS) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const video = webcamRef.current.video;

//             if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
//                 try {
//                     const canvas = document.createElement('canvas');
//                     canvas.width = video.videoWidth;
//                     canvas.height = video.videoHeight;
//                     const ctx = canvas.getContext('2d');
//                     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//                     const frameData = canvas.toDataURL('image/jpeg', 0.8); // Higher quality for premium
//                     frameCounterRef.current++;

//                     if (wsRef.current?.readyState === WebSocket.OPEN) {
//                         wsRef.current.send(JSON.stringify({
//                             type: 'frame',
//                             frame: frameData,
//                             timestamp: now,
//                             frameId: frameCounterRef.current,
//                             quality: 'premium'
//                         }));
//                     }

//                     lastSent = now;
//                 } catch (error) {
//                     console.log("❌ Error capturing frame:", error);
//                 }
//             }

//             animationRef.current = requestAnimationFrame(captureFrame);
//         };

//         captureFrame();
//     };

//     const generatePremiumCSV = () => {
//         const headers = [
//             "Timestamp", "Person Name", "Face ID", "Current Emotion", "Emotion Category",
//             "Emotion Confidence", "Confidence Level", "High Confidence", "Emotion Stability",
//             "Current Duration", "Total Duration", "Dominant Emotion", "Total Transitions",
//             "High Confidence Ratio", "Track Count", "Confirmed"
//         ];

//         const rows = trackingList.map(track => [
//             track.timestamp,
//             track.personName,
//             track.faceId,
//             track.current_emotion,
//             track.emotion_category,
//             (track.emotion_confidence * 100).toFixed(1) + '%',
//             getConfidenceLevel(track.emotion_confidence).text,
//             track.is_high_confidence ? 'YES' : 'NO',
//             (track.emotion_stability * 100).toFixed(1) + '%',
//             formatDuration(track.current_emotion_duration),
//             formatDuration(track.total_emotion_duration),
//             track.dominant_emotion,
//             track.total_transitions,
//             (track.high_confidence_ratio * 100).toFixed(1) + '%',
//             track.track_count,
//             track.confirmed ? 'YES' : 'NO'
//         ]);

//         const csvContent = [headers, ...rows].map(r => r.join(",")).join("\n");
//         return csvContent;
//     };

//     const downloadCSV = () => {
//         if (trackingList.length === 0) {
//             alert("No tracking data to download yet!");
//             return;
//         }

//         const csv = generatePremiumCSV();
//         const blob = new Blob([csv], { type: "text/csv" });
//         const url = window.URL.createObjectURL(blob);
//         const a = document.createElement("a");
//         a.href = url;
//         a.download = `premium_emotion_tracking_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
//         document.body.appendChild(a);
//         a.click();
//         document.body.removeChild(a);
//         window.URL.revokeObjectURL(url);
//     };

//     const clearAllRecords = () => {
//         if (trackingList.length === 0) {
//             alert("No records to clear!");
//             return;
//         }

//         const confirmClear = confirm(
//             `Are you sure you want to clear all ${trackingList.length} premium emotion records?`
//         );

//         if (confirmClear) {
//             setTrackingList([]);
//             setFaceTracks({});
//             setEmotionAnalytics({});
//             setSeenFaceIds(new Set());
//             alert("All premium records cleared!");
//         }
//     };

//     const getStatusColor = () => {
//         if (!connected) return "text-red-500";
//         return "text-green-500";
//     };

//     const getStatusEmoji = () => {
//         if (!connected) return "❌";
//         return "✅";
//     };

//     return (
//         <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gradient-to-br from-blue-50 to-purple-50 min-h-screen">
//             {/* Left Panel - Webcam & Premium Controls */}
//             <div className="flex flex-col items-center bg-white rounded-2xl shadow-xl p-6 w-full md:w-auto">
//                 <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
//                     Premium Emotion AI
//                 </h2>
//                 <p className={`text-sm mb-4 font-medium ${getStatusColor()} flex items-center gap-2`}>
//                     <span className="text-lg">{getStatusEmoji()}</span>
//                     {status}
//                 </p>

//                 {/* Webcam with premium overlay */}
//                 <div className="relative">
//                     <Webcam
//                         ref={webcamRef}
//                         width={540}
//                         height={405}
//                         className="rounded-xl border-4 border-purple-200 shadow-lg"
//                         audio={false}
//                         screenshotFormat="image/jpeg"
//                         videoConstraints={{
//                             facingMode: "user",
//                             width: 540,
//                             height: 405,
//                             frameRate: { ideal: 15, max: 20 }
//                         }}
//                         onUserMedia={() => console.log("✅ HD Webcam access granted")}
//                         onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
//                     />

//                     {/* Premium face bounding boxes */}
//                     {Object.entries(faceTracks).map(([trackId, track]) => {
//                         const analytics = emotionAnalytics[trackId] || {};
//                         const confidenceInfo = getConfidenceLevel(track.emotionConfidence);
//                         const stabilityInfo = getStabilityLevel(analytics.stability || 0);
                        
//                         return (
//                             <div
//                                 key={trackId}
//                                 className="absolute border-3 border-green-400 rounded-lg shadow-2xl"
//                                 style={{
//                                     left: `${track.bbox?.[0] || 0}px`,
//                                     top: `${track.bbox?.[1] || 0}px`,
//                                     width: `${track.bbox?.[2] || 0}px`,
//                                     height: `${track.bbox?.[3] || 0}px`,
//                                 }}
//                             >
//                                 <div className="absolute -top-10 left-0 bg-gradient-to-r from-green-500 to-blue-500 text-white text-sm px-3 py-2 rounded-lg shadow-lg">
//                                     <div className="flex items-center gap-2">
//                                         <span className="text-lg">{getEmotionEmoji(track.currentEmotion)}</span>
//                                         <span className="font-bold">{track.personName}</span>
//                                         {track.emotionChanged && (
//                                             <span className="ml-1 animate-pulse text-yellow-300">🔄</span>
//                                         )}
//                                         {track.isHighConfidence && (
//                                             <span className="ml-1 text-yellow-300">🎯</span>
//                                         )}
//                                     </div>
//                                     <div className="text-xs opacity-90">
//                                         {confidenceInfo.emoji} {(track.emotionConfidence * 100).toFixed(0)}% • 
//                                         {stabilityInfo.text} • {formatDuration(analytics.currentDuration || 0)}
//                                     </div>
//                                 </div>
//                             </div>
//                         );
//                     })}
//                 </div>

//                 {/* Premium Control Buttons */}
//                 <div className="mt-6 flex flex-col gap-3 w-full">
//                     <button
//                         onClick={downloadCSV}
//                         disabled={trackingList.length === 0}
//                         className={`px-6 py-3 rounded-xl transition-all flex items-center justify-center gap-3 font-semibold ${
//                             trackingList.length === 0
//                                 ? 'bg-gray-400 cursor-not-allowed'
//                                 : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl'
//                         }`}
//                     >
//                         <span className="text-xl">📊</span>
//                         <span>Export Premium Analytics ({trackingList.length})</span>
//                     </button>

//                     <button
//                         onClick={clearAllRecords}
//                         disabled={trackingList.length === 0}
//                         className={`px-6 py-3 rounded-xl transition-all flex items-center justify-center gap-3 font-semibold ${
//                             trackingList.length === 0
//                                 ? 'bg-gray-400 cursor-not-allowed'
//                                 : 'bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white shadow-lg hover:shadow-xl'
//                         }`}
//                     >
//                         <span className="text-xl">🗑️</span>
//                         <span>Clear All Records</span>
//                     </button>

//                     <button
//                         onClick={() => {
//                             if (wsRef.current) {
//                                 wsRef.current.close();
//                             }
//                             setConnected(false);
//                             setStatus("Reconnecting to premium server...");
//                             setTimeout(connectWebSocket, 1000);
//                         }}
//                         className="px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl hover:from-gray-700 hover:to-gray-800 transition-all flex items-center justify-center gap-3 font-semibold shadow-lg hover:shadow-xl"
//                     >
//                         <span className="text-xl">🔄</span>
//                         <span>Reconnect</span>
//                     </button>
//                 </div>

//                 {/* Premium Analytics Summary */}
//                 <div className="mt-6 w-full">
//                     <h3 className="font-bold text-gray-800 mb-3 text-lg">Live Analytics</h3>
//                     <div className="grid grid-cols-2 gap-3 text-sm">
//                         <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
//                             <div className="text-blue-800 font-semibold">High Confidence</div>
//                             <div className="text-2xl font-bold text-blue-600">
//                                 {Object.values(faceTracks).filter(t => t.isHighConfidence).length}
//                             </div>
//                         </div>
//                         <div className="bg-green-50 p-3 rounded-lg border border-green-200">
//                             <div className="text-green-800 font-semibold">Total Tracks</div>
//                             <div className="text-2xl font-bold text-green-600">
//                                 {Object.keys(faceTracks).length}
//                             </div>
//                         </div>
//                     </div>
//                 </div>
//             </div>

//             {/* Right Panel - Premium Emotion Analytics */}
//             <div className="bg-white w-full md:w-[500px] h-[600px] rounded-2xl shadow-xl p-6 overflow-y-auto border border-purple-200">
//                 <div className="flex justify-between items-center mb-4">
//                     <h3 className="text-xl font-bold text-gray-800 flex items-center gap-3">
//                         <span className="text-2xl">🎭</span>
//                         Premium Emotion Analytics
//                         <span className="text-xs bg-gradient-to-r from-purple-100 to-blue-100 text-purple-800 px-3 py-1 rounded-full">
//                             90%+ Accuracy
//                         </span>
//                     </h3>
//                     <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full font-semibold">
//                         {trackingList.length} entries
//                     </span>
//                 </div>

//                 {trackingList.length === 0 ? (
//                     <div className="text-center mt-20 text-gray-500">
//                         <div className="text-6xl mb-6">😊</div>
//                         <p className="text-lg font-semibold mb-2">Premium Emotion Tracking Ready</p>
//                         <p className="text-sm mb-6">High-confidence emotion detection will appear here</p>
//                         <div className="text-left text-sm space-y-3 max-w-xs mx-auto bg-blue-50 p-4 rounded-xl">
//                             <p className="font-bold text-blue-800">Premium Features:</p>
//                             <ul className="space-y-2">
//                                 <li className="flex items-center gap-2">🎯 90%+ Confidence Tracking</li>
//                                 <li className="flex items-center gap-2">📊 Emotion Stability Analysis</li>
//                                 <li className="flex items-center gap-2">⏱️ Duration & Frequency</li>
//                                 <li className="flex items-center gap-2">🔄 Transition Tracking</li>
//                                 <li className="flex items-center gap-2">💡 HD Camera Optimized</li>
//                             </ul>
//                         </div>
//                     </div>
//                 ) : (
//                     <>
//                         <ul className="space-y-4">
//                             {trackingList.map((track, index) => {
//                                 const confidenceInfo = getConfidenceLevel(track.emotion_confidence);
//                                 const stabilityInfo = getStabilityLevel(track.emotion_stability);
                                
//                                 return (
//                                     <li
//                                         key={index}
//                                         className={`flex justify-between items-start p-4 rounded-xl transition-all border-2 ${
//                                             track.is_high_confidence 
//                                                 ? 'bg-gradient-to-r from-green-50 to-blue-50 border-green-200 shadow-md' 
//                                                 : track.emotion_changed 
//                                                     ? 'bg-blue-50 border-blue-200' 
//                                                     : 'bg-gray-50 border-gray-200'
//                                         }`}
//                                     >
//                                         <div className="flex-1">
//                                             <div className="flex justify-between items-start mb-3">
//                                                 <div className="flex items-center gap-3">
//                                                     <span className="font-bold text-gray-900">{track.personName}</span>
//                                                     <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getEmotionColor(track.current_emotion)}`}>
//                                                         <span className="text-lg mr-1">{getEmotionEmoji(track.current_emotion)}</span>
//                                                         {track.current_emotion}
//                                                     </span>
//                                                 </div>
//                                                 <div className="flex gap-2">
//                                                     <span className={`px-2 py-1 rounded text-xs ${confidenceInfo.color}`}>
//                                                         {confidenceInfo.emoji} {(track.emotion_confidence * 100).toFixed(0)}%
//                                                     </span>
//                                                     {track.confirmed && (
//                                                         <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">✓</span>
//                                                     )}
//                                                 </div>
//                                             </div>
                                            
//                                             <p className="text-xs text-gray-500 mb-3">{track.timestamp}</p>
                                            
//                                             {/* Premium Analytics Row */}
//                                             <div className="grid grid-cols-2 gap-2 mb-3">
//                                                 <div className={`px-2 py-1 rounded text-xs ${stabilityInfo.color}`}>
//                                                     Stability: {stabilityInfo.text}
//                                                 </div>
//                                                 <div className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs">
//                                                     ⏱️ {formatDuration(track.current_emotion_duration)}
//                                                 </div>
//                                                 <div className="bg-orange-100 text-orange-800 px-2 py-1 rounded text-xs">
//                                                     🔄 {track.total_transitions} transitions
//                                                 </div>
//                                                 <div className="bg-cyan-100 text-cyan-800 px-2 py-1 rounded text-xs">
//                                                     📈 {(track.high_confidence_ratio * 100).toFixed(0)}% reliable
//                                                 </div>
//                                             </div>

//                                             {/* Context Information */}
//                                             {track.context && (track.context.estimated_age || track.context.gender) && (
//                                                 <div className="bg-gray-100 p-2 rounded-lg text-xs">
//                                                     <div className="font-semibold text-gray-700">Context:</div>
//                                                     <div className="flex gap-4 mt-1">
//                                                         {track.context.estimated_age && (
//                                                             <span>Age: ~{track.context.estimated_age}</span>
//                                                         )}
//                                                         {track.context.gender && (
//                                                             <span>Gender: {track.context.gender}</span>
//                                                         )}
//                                                     </div>
//                                                 </div>
//                                             )}

//                                             {/* Recent Trend */}
//                                             {track.recent_trend && track.recent_trend.length > 0 && (
//                                                 <div className="mt-2">
//                                                     <div className="text-xs text-gray-600 font-semibold">Recent Trend:</div>
//                                                     <div className="flex gap-1 mt-1">
//                                                         {track.recent_trend.slice(-5).map((emotion, idx) => (
//                                                             <span key={idx} className="text-xs bg-white px-2 py-1 rounded border">
//                                                                 {getEmotionEmoji(emotion)}
//                                                             </span>
//                                                         ))}
//                                                     </div>
//                                                 </div>
//                                             )}
//                                         </div>
//                                     </li>
//                                 );
//                             })}
//                         </ul>
                        
//                         <div className="mt-6 pt-4 border-t border-gray-200">
//                             <button
//                                 onClick={clearAllRecords}
//                                 className="w-full px-6 py-3 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white rounded-xl transition-all flex items-center justify-center gap-3 font-semibold shadow-lg"
//                             >
//                                 <span className="text-xl">🗑️</span>
//                                 <span>Clear All {trackingList.length} Records</span>
//                             </button>
//                         </div>
//                     </>
//                 )}
//             </div>
//         </div>
//     );
// }
// import { useEffect, useRef, useState } from "react";
// import Webcam from "react-webcam";

// export default function EmotionTrackerPython() {
//     const webcamRef = useRef(null);
//     const [status, setStatus] = useState("Idle");
//     const [emotionList, setEmotionList] = useState([]);
//     const [connected, setConnected] = useState(false);
//     const [faceTracks, setFaceTracks] = useState({});
//     const wsRef = useRef(null);
//     const animationRef = useRef(null);
//     const frameCounterRef = useRef(0);
//     const MAX_ENTRIES = 1000; // Keep last 1,000 entries

//     // Utility functions
//     const checkEmotionChange = (previousEmotions, currentDetection) => {
//         if (!previousEmotions || previousEmotions.length === 0) {
//             return false;
//         }

//         // Find the most recent emotion for this face
//         const lastEmotionForFace = previousEmotions.find(
//             emotion => emotion.faceId === currentDetection.face_id
//         );

//         if (!lastEmotionForFace) {
//             return false;
//         }

//         return lastEmotionForFace.emotion !== currentDetection.emotion;
//     };

//     const checkNewFace = (previousEmotions, currentFaceId) => {
//         if (!previousEmotions || previousEmotions.length === 0) {
//             return true;
//         }

//         return !previousEmotions.some(emotion => emotion.faceId === currentFaceId);
//     };

//     useEffect(() => {
//         connectWebSocket();

//         return () => {
//             if (wsRef.current) {
//                 wsRef.current.close();
//             }
//             if (animationRef.current) {
//                 cancelAnimationFrame(animationRef.current);
//             }
//         };
//     }, []);

//     const connectWebSocket = () => {
//         setStatus("Connecting to AI server...");

//         try {
//             const ws = new WebSocket("ws://localhost:8765");
//             wsRef.current = ws;

//             ws.onopen = () => {
//                 console.log("✅ WebSocket connected successfully");
//                 setConnected(true);
//                 setStatus("Connected! Starting detection...");
//                 startFrameCapture();
//             };

//             ws.onmessage = (event) => {
//                 try {
//                     const data = JSON.parse(event.data);

//                     if (data.type === 'detection_results') {
//                         const newFaceTracks = {};

//                         data.detections.forEach(detection => {
//                             const faceId = detection.face_id;
//                             console.log("Processing detection for face ID:", faceId, "detection", detection);
//                             newFaceTracks[faceId] = {
//                                 id: faceId,
//                                 personName: `Person_${detection.face_id}`,
//                                 bbox: detection.bbox || [0, 0, 0, 0],
//                                 currentEmotion: detection.emotion || 'neutral',
//                                 confidence: detection.confidence || 0.8,
//                                 gender: detection.gender || 'unknown',
//                                 genderConfidence: detection.gender_confidence || 0.5,
//                                 detectionCount: detection.track_count || 1,
//                                 lastSeen: Date.now()
//                             };
//                         });

//                         setFaceTracks(newFaceTracks);

//                         // Add to emotion list
//                         if (data.detections.length > 0) {
//                             setEmotionList(prev => {
//                                 const newEntries = data.detections.map(detection => ({
//                                     personName: detection.person_name || `Person_${detection.face_id}`,
//                                     faceId: detection.face_id,
//                                     emotion: detection.emotion || 'neutral',
//                                     confidence: detection.confidence || 0.8,
//                                     gender: detection.gender || 'unknown',
//                                     genderConfidence: detection.gender_confidence || 0.5,
//                                     bbox: detection.bbox || [],
//                                     timestamp: detection.timestamp || new Date().toISOString(),
//                                     track_count: detection.track_count || 1,
//                                     similarity_score: detection.similarity_score || 0,
//                                     emotionChanged: checkEmotionChange(prev, detection),
//                                     isNewFace: checkNewFace(prev, detection.face_id)
//                                 }));

//                                 return [...newEntries, ...prev].slice(0, MAX_ENTRIES);
//                             });
//                         }
//                     }
//                 } catch (error) {
//                     console.error('Error processing WebSocket message:', error);
//                 }
//             };

//             ws.onclose = (event) => {
//                 console.log("📞 WebSocket disconnected:", event.code, event.reason);
//                 setConnected(false);
//                 setStatus(`Disconnected (${event.code}). Retrying in 3s...`);

//                 // Clear any existing timeouts to prevent multiple retries
//                 setTimeout(() => {
//                     if (!connected) {
//                         connectWebSocket();
//                     }
//                 }, 3000);
//             };

//             ws.onerror = (error) => {
//                 console.log("❌ WebSocket error:", error);
//                 setStatus("Connection error - Check if Python server is running");
//                 setConnected(false);
//             };
//         } catch (error) {
//             console.log("❌ Failed to create WebSocket:", error);
//             setStatus("Failed to connect - Retrying...");
//             setTimeout(connectWebSocket, 3000);
//         }
//     };

//     const startFrameCapture = () => {
//         let lastSent = 0;
//         const FPS = 10; // Increased to 10 FPS for better responsiveness

//         const captureFrame = () => {
//             // Only capture if connected and webcam is ready
//             if (!connected || !webcamRef.current?.video || wsRef.current?.readyState !== WebSocket.OPEN) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const now = Date.now();
//             if (now - lastSent < 1000 / FPS) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const video = webcamRef.current.video;

//             // Check if video is ready and has data
//             if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
//                 try {
//                     const canvas = document.createElement('canvas');
//                     canvas.width = video.videoWidth;
//                     canvas.height = video.videoHeight;
//                     const ctx = canvas.getContext('2d');
//                     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//                     // Reduce quality to minimize data size
//                     const frameData = canvas.toDataURL('image/jpeg', 0.7);
//                     frameCounterRef.current++;

//                     // Send frame to backend
//                     if (wsRef.current?.readyState === WebSocket.OPEN) {
//                         wsRef.current.send(JSON.stringify({
//                             type: 'frame',
//                             frame: frameData,
//                             timestamp: now,
//                             frameId: frameCounterRef.current
//                         }));
//                     }

//                     lastSent = now;
//                 } catch (error) {
//                     console.log("❌ Error capturing frame:", error);
//                 }
//             }

//             animationRef.current = requestAnimationFrame(captureFrame);
//         };

//         captureFrame();
//     };

//     const generateEnhancedCSV = () => {
//         const headers = [
//             "Timestamp", "Person Name", "Face ID", "Emotion", "Confidence",
//             "Gender", "Gender Confidence", "Bounding Box", "Emotion Changed",
//             "New Face", "Track Count", "Similarity Score"
//         ];

//         const rows = emotionList.map(e => [
//             e.timestamp,
//             e.personName || `Person_${e.faceId}`,
//             e.faceId,
//             e.emotion,
//             e.confidence,
//             e.gender || 'unknown',
//             e.genderConfidence || 'N/A',
//             e.bbox && e.bbox.length === 4 ? `[${e.bbox.join(',')}]` : '[]',
//             e.emotionChanged ? 'YES' : 'NO',
//             e.isNewFace ? 'YES' : 'NO',
//             e.track_count || '0',
//             e.similarity_score || 'N/A'
//         ]);

//         const csvContent = [headers, ...rows].map(r => r.join(",")).join("\n");
//         return csvContent;
//     };

//     const downloadCSV = () => {
//         if (emotionList.length === 0) {
//             alert("No emotion data to download yet!");
//             return;
//         }

//         // Warn for very large datasets
//         if (emotionList.length > 1000) {
//             const shouldContinue = confirm(
//                 `You are about to download ${emotionList.length} entries. This may create a large file. Continue?`
//             );
//             if (!shouldContinue) return;
//         }

//         const csv = generateEnhancedCSV();
//         const blob = new Blob([csv], { type: "text/csv" });
//         const url = window.URL.createObjectURL(blob);
//         const a = document.createElement("a");
//         a.href = url;
//         a.download = `emotion_data_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
//         document.body.appendChild(a);
//         a.click();
//         document.body.removeChild(a);
//         window.URL.revokeObjectURL(url);
//     };

//     const clearAllRecords = () => {
//         if (emotionList.length === 0) {
//             alert("No records to clear!");
//             return;
//         }

//         const confirmClear = confirm(
//             `Are you sure you want to clear all ${emotionList.length} records? This action cannot be undone.`
//         );

//         if (confirmClear) {
//             setEmotionList([]);
//             setFaceTracks({});
//             alert("All records have been cleared!");
//         }
//     };

//     const clearFaceTracks = () => {
//         if (Object.keys(faceTracks).length === 0) {
//             alert("No active face tracks to clear!");
//             return;
//         }

//         const confirmClear = confirm(
//             `Clear ${Object.keys(faceTracks).length} active face tracks? This will reset face IDs.`
//         );

//         if (confirmClear) {
//             setFaceTracks({});
//             alert("Face tracks cleared!");
//         }
//     };

//     const getStatusColor = () => {
//         if (!connected) return "text-red-500";
//         return "text-green-500";
//     };

//     const getStatusEmoji = () => {
//         if (!connected) return "❌";
//         return "✅";
//     };

//     return (
//         <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
//             {/* Left Panel - Webcam & Controls */}
//             <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
//                 <h2 className="text-2xl font-bold mb-2 text-gray-700">AI Emotion & Gender Tracker</h2>
//                 <p className={`text-sm mb-3 font-medium ${getStatusColor()}`}>
//                     {getStatusEmoji()} {status}
//                 </p>

//                 {/* Webcam with face overlay */}
//                 <div className="relative">
//                     <Webcam
//                         ref={webcamRef}
//                         width={480}
//                         height={360}
//                         className="rounded-lg border-2 border-gray-300"
//                         audio={false}
//                         screenshotFormat="image/jpeg"
//                         videoConstraints={{
//                             facingMode: "user",
//                             width: 480,
//                             height: 360,
//                             frameRate: { ideal: 10, max: 15 }
//                         }}
//                         onUserMedia={() => console.log("✅ Webcam access granted")}
//                         onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
//                     />

//                     {/* Face bounding boxes overlay */}
//                     {Object.entries(faceTracks).map(([trackId, track]) => (
//                         <div
//                             key={trackId}
//                             className="absolute border-2 border-green-400 rounded-lg"
//                             style={{
//                                 left: `${track.bbox?.[0] || 0}px`,
//                                 top: `${track.bbox?.[1] || 0}px`,
//                                 width: `${track.bbox?.[2] || 0}px`,
//                                 height: `${track.bbox?.[3] || 0}px`,
//                             }}
//                         >
//                             <div className="absolute -top-8 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
//                                 {track.personName} - {track.currentEmotion}
//                                 {track.gender !== 'unknown' && (
//                                     <span className="ml-1 text-xs">({track.gender})</span>
//                                 )}
//                             </div>
//                         </div>
//                     ))}
//                 </div>

//                 {/* Control Buttons */}
//                 <div className="mt-4 flex flex-col gap-3 w-full">
//                     {/* Download CSV Button */}
//                     <button
//                         onClick={downloadCSV}
//                         disabled={emotionList.length === 0}
//                         className={`px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
//                             emotionList.length === 0
//                                 ? 'bg-gray-400 cursor-not-allowed'
//                                 : 'bg-blue-600 hover:bg-blue-700 text-white'
//                         }`}
//                     >
//                         <span>📥</span>
//                         <span>Download CSV ({emotionList.length} entries)</span>
//                     </button>

//                     {/* Clear Records Button */}
//                     <button
//                         onClick={clearAllRecords}
//                         disabled={emotionList.length === 0}
//                         className={`px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
//                             emotionList.length === 0
//                                 ? 'bg-gray-400 cursor-not-allowed'
//                                 : 'bg-red-600 hover:bg-red-700 text-white'
//                         }`}
//                     >
//                         <span>🗑️</span>
//                         <span>Clear All Records</span>
//                     </button>

//                     {/* Clear Face Tracks Button */}
//                     <button
//                         onClick={clearFaceTracks}
//                         disabled={Object.keys(faceTracks).length === 0}
//                         className={`px-4 py-2 rounded-lg transition flex items-center justify-center gap-2 ${
//                             Object.keys(faceTracks).length === 0
//                                 ? 'bg-gray-400 cursor-not-allowed'
//                                 : 'bg-orange-600 hover:bg-orange-700 text-white'
//                         }`}
//                     >
//                         <span>👤</span>
//                         <span>Clear Face Tracks ({Object.keys(faceTracks).length})</span>
//                     </button>

//                     {/* Reconnect Button */}
//                     <button
//                         onClick={() => {
//                             if (wsRef.current) {
//                                 wsRef.current.close();
//                             }
//                             setConnected(false);
//                             setStatus("Reconnecting...");
//                             setTimeout(connectWebSocket, 1000);
//                         }}
//                         className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition flex items-center justify-center gap-2"
//                     >
//                         <span>🔄</span>
//                         <span>Reconnect</span>
//                     </button>
//                 </div>

//                 {/* Enhanced Face Tracking Info */}
//                 <div className="mt-4 w-full">
//                     <h3 className="font-semibold text-gray-700 mb-2">
//                         Active Faces: {Object.keys(faceTracks).length}
//                     </h3>
//                     <div className="text-xs text-gray-600 space-y-1 max-h-32 overflow-y-auto">
//                         {Object.entries(faceTracks).length === 0 ? (
//                             <div className="text-center py-4 text-gray-400">
//                                 <p>No faces detected yet...</p>
//                                 <p className="text-xs mt-1">Make sure you're facing the camera</p>
//                             </div>
//                         ) : (
//                             Object.entries(faceTracks).map(([trackId, track]) => (
//                                 <div key={trackId} className="flex justify-between items-center p-2 bg-gray-50 rounded border">
//                                     <div className="flex items-center gap-2">
//                                         <span className="font-medium">{track.personName}</span>
//                                         <span className={`px-2 py-1 rounded text-xs capitalize ${
//                                             track.currentEmotion === 'happy' ? 'bg-green-100 text-green-800' :
//                                             track.currentEmotion === 'sad' ? 'bg-blue-100 text-blue-800' :
//                                             track.currentEmotion === 'angry' ? 'bg-red-100 text-red-800' :
//                                             'bg-gray-100 text-gray-800'
//                                         }`}>
//                                             {track.currentEmotion}
//                                         </span>
//                                         {track.gender !== 'unknown' && (
//                                             <span className={`text-xs px-2 py-1 rounded ${
//                                                 track.gender === 'man' ? 'bg-blue-100 text-blue-800' :
//                                                 'bg-pink-100 text-pink-800'
//                                             }`}>
//                                                 {track.gender}
//                                             </span>
//                                         )}
//                                     </div>
//                                     <div className="flex gap-1 text-xs">
//                                         <span className="bg-purple-100 px-1 rounded">#{track.detectionCount}</span>
//                                         <span className="bg-orange-100 px-1 rounded">{Math.round(track.confidence * 100)}%</span>
//                                     </div>
//                                 </div>
//                             ))
//                         )}
//                     </div>
//                 </div>
//             </div>

//             {/* Right Panel - Enhanced Emotion Log */}
//             <div className="bg-white w-full md:w-96 h-[500px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
//                 <div className="flex justify-between items-center mb-3">
//                     <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
//                         🧠 Emotion Log
//                         <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
//                             DeepFace AI
//                         </span>
//                     </h3>
//                     <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
//                         {emotionList.length} entries
//                     </span>
//                 </div>

//                 {emotionList.length === 0 ? (
//                     <div className="text-center mt-20 text-gray-400">
//                         <div className="text-4xl mb-4">📊</div>
//                         <p className="text-sm font-medium mb-2">No emotion data yet</p>
//                         <p className="text-xs mb-4">Emotion detections will appear here</p>
//                         <div className="text-left text-xs space-y-2 max-w-xs mx-auto">
//                             <p className="font-medium">Powered by DeepFace AI:</p>
//                             <ul className="space-y-1">
//                                 <li>• Real-time face recognition</li>
//                                 <li>• Emotion & gender detection</li>
//                                 <li>• Multi-person analysis</li>
//                                 <li>• High-confidence tracking</li>
//                             </ul>
//                         </div>
//                     </div>
//                 ) : (
//                     <>
//                         <ul className="space-y-2">
//                             {emotionList.map((e, index) => (
//                                 <li
//                                     key={index}
//                                     className={`flex justify-between items-start p-3 rounded-lg transition-all ${
//                                         e.emotionChanged ? 'bg-green-50 border border-green-200' :
//                                         e.isNewFace ? 'bg-yellow-50 border border-yellow-200' :
//                                         'bg-gray-50 border border-gray-200'
//                                     }`}
//                                 >
//                                     <div className="flex-1">
//                                         <div className="flex justify-between items-start mb-1">
//                                             <div className="flex items-center gap-2">
//                                                 <span className="font-medium capitalize text-gray-800">{e.emotion}</span>
//                                                 {e.gender && e.gender !== 'unknown' && (
//                                                     <span className={`text-xs px-2 py-1 rounded ${
//                                                         e.gender === 'man' ? 'bg-blue-100 text-blue-800' :
//                                                         'bg-pink-100 text-pink-800'
//                                                     }`}>
//                                                         {e.gender}
//                                                     </span>
//                                                 )}
//                                             </div>
//                                             <div className="flex gap-1 text-xs">
//                                                 <span className="bg-blue-100 px-2 py-1 rounded">{e.confidence}</span>
//                                                 {e.genderConfidence && e.genderConfidence > 0.5 && (
//                                                     <span className="bg-purple-100 px-2 py-1 rounded">{e.genderConfidence}</span>
//                                                 )}
//                                             </div>
//                                         </div>
//                                         <p className="text-xs text-gray-500 mb-2">{e.timestamp}</p>
//                                         <div className="flex gap-2 flex-wrap">
//                                             <span className="text-xs bg-gray-200 px-2 py-1 rounded">{e.personName}</span>
//                                             <span className="text-xs bg-purple-200 px-2 py-1 rounded">Track: {e.track_count}</span>
//                                             {e.similarity_score && (
//                                                 <span className="text-xs bg-blue-200 px-2 py-1 rounded">Similarity: {e.similarity_score}</span>
//                                             )}
//                                             {e.emotionChanged && (
//                                                 <span className="text-xs bg-green-200 px-2 py-1 rounded">Emotion Changed</span>
//                                             )}
//                                             {e.isNewFace && (
//                                                 <span className="text-xs bg-yellow-200 px-2 py-1 rounded">New Face</span>
//                                             )}
//                                         </div>
//                                     </div>
//                                 </li>
//                             ))}
//                         </ul>
                        
//                         {/* Clear button at bottom when there are entries */}
//                         <div className="mt-4 pt-4 border-t border-gray-200">
//                             <button
//                                 onClick={clearAllRecords}
//                                 className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition flex items-center justify-center gap-2"
//                             >
//                                 <span>🗑️</span>
//                                 <span>Clear All {emotionList.length} Records</span>
//                             </button>
//                         </div>
//                     </>
//                 )}
//             </div>
//         </div>
//     );
// }
// "use client";
// #V1 working  on PRODUCTION Emotion Detection Server with Advanced Face Tracking
// import { useEffect, useRef, useState } from "react";
// import Webcam from "react-webcam";

// export default function EmotionTrackerPython() {
//     const webcamRef = useRef(null);
//     const [status, setStatus] = useState("Idle");
//     const [emotionList, setEmotionList] = useState([]);
//     const [connected, setConnected] = useState(false);
//     const [faceTracks, setFaceTracks] = useState({});
//     const wsRef = useRef(null);
//     const animationRef = useRef(null);
//     const frameCounterRef = useRef(0);
//     const MAX_ENTRIES = 1000; // Keep last 10,000 entries

//     // Utility functions
//     const checkEmotionChange = (previousEmotions, currentDetection) => {
//         if (!previousEmotions || previousEmotions.length === 0) {
//             return false;
//         }

//         // Find the most recent emotion for this face
//         const lastEmotionForFace = previousEmotions.find(
//             emotion => emotion.faceId === currentDetection.face_id
//         );

//         if (!lastEmotionForFace) {
//             return false;
//         }

//         return lastEmotionForFace.emotion !== currentDetection.emotion;
//     };

//     const checkNewFace = (previousEmotions, currentFaceId) => {
//         if (!previousEmotions || previousEmotions.length === 0) {
//             return true;
//         }

//         return !previousEmotions.some(emotion => emotion.faceId === currentFaceId);
//     };

//     useEffect(() => {
//         connectWebSocket();

//         return () => {
//             if (wsRef.current) {
//                 wsRef.current.close();
//             }
//             if (animationRef.current) {
//                 cancelAnimationFrame(animationRef.current);
//             }
//         };
//     }, []);

//     const connectWebSocket = () => {
//         setStatus("Connecting to AI server...");

//         try {
//             const ws = new WebSocket("ws://localhost:8765");
//             wsRef.current = ws;

//             ws.onopen = () => {
//                 console.log("✅ WebSocket connected successfully");
//                 setConnected(true);
//                 setStatus("Connected! Starting detection...");
//                 startFrameCapture();
//             };

//             ws.onmessage = (event) => {
//                 try {
//                     const data = JSON.parse(event.data);

//                     if (data.type === 'detection_results') {
//                         const newFaceTracks = {};

//                         data.detections.forEach(detection => {
//                             const faceId = detection.face_id;

//                             newFaceTracks[faceId] = {
//                                 id: faceId,
//                                 personName: detection.person_name || `Person_${detection.face_id}`,
//                                 bbox: detection.bbox || [0, 0, 0, 0],
//                                 currentEmotion: detection.emotion || 'neutral',
//                                 confidence: detection.confidence || 0.8,
//                                 gender: detection.gender || 'unknown',
//                                 genderConfidence: detection.gender_confidence || 0.5,
//                                 detectionCount: detection.track_count || 1,
//                                 lastSeen: Date.now()
//                             };
//                         });

//                         setFaceTracks(newFaceTracks);

//                         // Add to emotion list
//                         if (data.detections.length > 0) {
//                             setEmotionList(prev => {
//                                 const newEntries = data.detections.map(detection => ({
//                                     personName: detection.person_name || `Person_${detection.face_id}`,
//                                     faceId: detection.face_id,
//                                     emotion: detection.emotion || 'neutral',
//                                     confidence: detection.confidence || 0.8,
//                                     gender: detection.gender || 'unknown',
//                                     genderConfidence: detection.gender_confidence || 0.5,
//                                     bbox: detection.bbox || [],
//                                     timestamp: detection.timestamp || new Date().toISOString(),
//                                     track_count: detection.track_count || 1,
//                                     similarity_score: detection.similarity_score || 0,
//                                     emotionChanged: checkEmotionChange(prev, detection),
//                                     isNewFace: checkNewFace(prev, detection.face_id)
//                                 }));

//                                 return [...newEntries, ...prev].slice(0, MAX_ENTRIES);
//                             });
//                         }
//                     }
//                 } catch (error) {
//                     console.error('Error processing WebSocket message:', error);
//                 }
//             };

//             ws.onclose = (event) => {
//                 console.log("📞 WebSocket disconnected:", event.code, event.reason);
//                 setConnected(false);
//                 setStatus(`Disconnected (${event.code}). Retrying in 3s...`);

//                 // Clear any existing timeouts to prevent multiple retries
//                 setTimeout(() => {
//                     if (!connected) {
//                         connectWebSocket();
//                     }
//                 }, 3000);
//             };

//             ws.onerror = (error) => {
//                 console.log("❌ WebSocket error:", error);
//                 setStatus("Connection error - Check if Python server is running");
//                 setConnected(false);
//             };
//         } catch (error) {
//             console.log("❌ Failed to create WebSocket:", error);
//             setStatus("Failed to connect - Retrying...");
//             setTimeout(connectWebSocket, 3000);
//         }
//     };

//     const startFrameCapture = () => {
//         let lastSent = 0;
//         const FPS = 10; // Reduced to 3 FPS to avoid overwhelming the backend

//         const captureFrame = () => {
//             // Only capture if connected and webcam is ready
//             if (!connected || !webcamRef.current?.video || wsRef.current?.readyState !== WebSocket.OPEN) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const now = Date.now();
//             if (now - lastSent < 1000 / FPS) {
//                 animationRef.current = requestAnimationFrame(captureFrame);
//                 return;
//             }

//             const video = webcamRef.current.video;

//             // Check if video is ready and has data
//             if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
//                 try {
//                     const canvas = document.createElement('canvas');
//                     canvas.width = video.videoWidth;
//                     canvas.height = video.videoHeight;
//                     const ctx = canvas.getContext('2d');
//                     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//                     // Reduce quality to minimize data size
//                     const frameData = canvas.toDataURL('image/jpeg', 0.7);
//                     frameCounterRef.current++;

//                     // Send frame to backend
//                     if (wsRef.current?.readyState === WebSocket.OPEN) {
//                         wsRef.current.send(JSON.stringify({
//                             type: 'frame',
//                             frame: frameData,
//                             timestamp: now,
//                             frameId: frameCounterRef.current
//                         }));
//                     }

//                     lastSent = now;
//                 } catch (error) {
//                     console.log("❌ Error capturing frame:", error);
//                 }
//             }

//             animationRef.current = requestAnimationFrame(captureFrame);
//         };

//         captureFrame();
//     };

//     // const processDetections = (detections) => {
//     //     if (!detections || detections.length === 0) {
//     //         return;
//     //     }

//     //     const newEntries = [];
//     //     const updatedTracks = { ...faceTracks };

//     //     detections.forEach(detection => {
//     //         const { face_id, emotion, confidence, bbox, timestamp, track_count, confirmed } = detection;

//     //         // Only process confirmed tracks with valid emotions
//     //         if (!confirmed || !emotion || confidence < 0.3) {
//     //             return;
//     //         }

//     //         // Update face tracks
//     //         if (!updatedTracks[face_id]) {
//     //             updatedTracks[face_id] = {
//     //                 id: face_id,
//     //                 firstSeen: new Date().toLocaleTimeString(),
//     //                 emotionHistory: [],
//     //                 currentEmotion: emotion,
//     //                 detectionCount: 0,
//     //                 lastSeen: new Date().toLocaleTimeString(),
//     //                 trackCount: track_count,
//     //                 confirmed: confirmed
//     //             };
//     //         }

//     //         const track = updatedTracks[face_id];
//     //         track.detectionCount++;
//     //         track.currentEmotion = emotion;
//     //         track.lastSeen = new Date().toLocaleTimeString();
//     //         track.trackCount = track_count;

//     //         // Only add entry if emotion changed OR it's been more than 3 seconds
//     //         const lastEmotion = track.emotionHistory[0]?.emotion;
//     //         const emotionChanged = !lastEmotion || lastEmotion !== emotion;
//     //         const timeSinceLastEntry = track.emotionHistory[0] ?
//     //             (new Date() - new Date(track.emotionHistory[0].timestamp)) / 1000 : 999;

//     //         // More conservative: only log significant changes or every 3+ seconds
//     //         if ((emotionChanged && confidence > 0.5) || timeSinceLastEntry > 3) {
//     //             const newEntry = {
//     //                 timestamp: new Date().toLocaleTimeString(),
//     //                 emotion,
//     //                 confidence: typeof confidence === 'number' ? confidence.toFixed(3) : confidence,
//     //                 faceId: `user_${face_id}`,
//     //                 bbox: bbox || [],
//     //                 emotionChanged,
//     //                 isNewFace: track.detectionCount === 1,
//     //                 trackCount: track_count,
//     //                 confirmed: confirmed
//     //             };

//     //             newEntries.push(newEntry);
//     //             track.emotionHistory.unshift({
//     //                 emotion,
//     //                 confidence: typeof confidence === 'number' ? confidence : parseFloat(confidence),
//     //                 timestamp: new Date().toLocaleTimeString()
//     //             });

//     //             // Keep only last 15 emotions per face
//     //             if (track.emotionHistory.length > 15) {
//     //                 track.emotionHistory = track.emotionHistory.slice(0, 15);
//     //             }

//     //             console.log(`✅ Track ${face_id} (${track_count} frames): ${emotion} (${confidence})`);
//     //         }
//     //     });

//     //     // Update state with new entries
//     //     if (newEntries.length > 0) {
//     //         setEmotionList(prev => [...newEntries, ...prev.slice(0, 150)]); // Keep fewer entries
//     //     }
//     //     setFaceTracks(updatedTracks);
//     // };

//     const generateEnhancedCSV = () => {
//         const headers = [
//             "Timestamp", "Person Name", "Face ID", "Emotion", "Confidence",
//             "Gender", "Gender Confidence", "Bounding Box", "Emotion Changed",
//             "New Face", "Track Count", "Similarity Score"
//         ];

//         const rows = emotionList.map(e => [
//             e.timestamp,
//             e.personName || `Person_${e.faceId}`,
//             e.faceId,
//             e.emotion,
//             e.confidence,
//             e.gender || 'unknown',
//             e.genderConfidence || 'N/A',
//             e.bbox && e.bbox.length === 4 ? `[${e.bbox.join(',')}]` : '[]',
//             e.emotionChanged ? 'YES' : 'NO',
//             e.isNewFace ? 'YES' : 'NO',
//             e.track_count || '0',
//             e.similarity_score || 'N/A'
//         ]);

//         const csvContent = [headers, ...rows].map(r => r.join(",")).join("\n");
//         return csvContent;
//     };

//     const downloadCSV = () => {
//         if (emotionList.length === 0) {
//             alert("No emotion data to download yet!");
//             return;
//         }

//         const csv = generateEnhancedCSV();
//         const blob = new Blob([csv], { type: "text/csv" });
//         const url = window.URL.createObjectURL(blob);
//         const a = document.createElement("a");
//         a.href = url;
//         a.download = `emotion_log_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
//         document.body.appendChild(a);
//         a.click();
//         document.body.removeChild(a);
//         window.URL.revokeObjectURL(url);
//     };

//     const getStatusColor = () => {
//         if (!connected) return "text-red-500";
//         return "text-green-500";
//     };

//     const getStatusEmoji = () => {
//         if (!connected) return "❌";
//         return "✅";
//     };

//     return (
//         <div className="flex flex-col md:flex-row gap-6 items-start justify-center p-6 bg-gray-50 min-h-screen">
//             {/* Left Panel - Webcam & Controls */}
//             <div className="flex flex-col items-center bg-white rounded-2xl shadow-md p-5">
//                 <h2 className="text-2xl font-bold mb-2 text-gray-700">AI Emotion & Gender Tracker</h2>
//                 <p className={`text-sm mb-3 font-medium ${getStatusColor()}`}>
//                     {getStatusEmoji()} {status}
//                 </p>

//                 {/* Webcam with face overlay */}
//                 <div className="relative">
//                     <Webcam
//                         ref={webcamRef}
//                         width={480}
//                         height={360}
//                         className="rounded-lg border-2 border-gray-300"
//                         audio={false}
//                         screenshotFormat="image/jpeg"
//                         videoConstraints={{
//                             facingMode: "user",
//                             width: 480,
//                             height: 360,
//                             frameRate: { ideal: 10, max: 15 }
//                         }}
//                         onUserMedia={() => console.log("✅ Webcam access granted")}
//                         onUserMediaError={(error) => console.log("❌ Webcam error:", error)}
//                     />

//                     {/* Face bounding boxes overlay */}
//                     {Object.entries(faceTracks).map(([trackId, track]) => (
//                         <div
//                             key={trackId}
//                             className="absolute border-2 border-green-400 rounded-lg"
//                             style={{
//                                 left: `${track.bbox?.[0] || 0}px`,
//                                 top: `${track.bbox?.[1] || 0}px`,
//                                 width: `${track.bbox?.[2] || 0}px`,
//                                 height: `${track.bbox?.[3] || 0}px`,
//                             }}
//                         >
//                             <div className="absolute -top-8 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
//                                 {track.personName} - {track.currentEmotion}
//                                 {track.gender !== 'unknown' && (
//                                     <span className="ml-1 text-xs">({track.gender})</span>
//                                 )}
//                             </div>
//                         </div>
//                     ))}
//                 </div>

//                 <div className="mt-4 flex flex-col gap-3 w-full">
//                     <button
//                         onClick={downloadCSV}
//                         disabled={emotionList.length === 0}
//                         className={`px-4 py-2 rounded-lg transition ${emotionList.length === 0
//                             ? 'bg-gray-400 cursor-not-allowed'
//                             : 'bg-blue-600 hover:bg-blue-700 text-white'
//                             }`}
//                     >
//                         📥 Download CSV ({emotionList.length} entries)
//                     </button>

//                     <button
//                         onClick={() => {
//                             if (wsRef.current) {
//                                 wsRef.current.close();
//                             }
//                             setConnected(false);
//                             setStatus("Reconnecting...");
//                             setTimeout(connectWebSocket, 1000);
//                         }}
//                         className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
//                     >
//                         🔄 Reconnect
//                     </button>
//                 </div>

//                 {/* Enhanced Face Tracking Info */}
//                 <div className="mt-4 w-full">
//                     <h3 className="font-semibold text-gray-700 mb-2">
//                         Active Faces: {Object.keys(faceTracks).length}
//                     </h3>
//                     <div className="text-xs text-gray-600 space-y-1 max-h-32 overflow-y-auto">
//                         {Object.entries(faceTracks).length === 0 ? (
//                             <p className="text-gray-400">No faces detected yet...</p>
//                         ) : (
//                             Object.entries(faceTracks).map(([trackId, track]) => (
//                                 <div key={trackId} className="flex justify-between items-center p-2 bg-gray-50 rounded border">
//                                     <div className="flex items-center gap-2">
//                                         <span className="font-medium">{track.personName}</span>
//                                         <span className={`px-2 py-1 rounded text-xs capitalize ${track.currentEmotion === 'happy' ? 'bg-green-100 text-green-800' :
//                                             track.currentEmotion === 'sad' ? 'bg-blue-100 text-blue-800' :
//                                                 track.currentEmotion === 'angry' ? 'bg-red-100 text-red-800' :
//                                                     'bg-gray-100 text-gray-800'
//                                             }`}>
//                                             {track.currentEmotion}
//                                         </span>
//                                         {track.gender !== 'unknown' && (
//                                             <span className={`text-xs px-2 py-1 rounded ${track.gender === 'man' ? 'bg-blue-100 text-blue-800' :
//                                                 'bg-pink-100 text-pink-800'
//                                                 }`}>
//                                                 {track.gender}
//                                             </span>
//                                         )}
//                                     </div>
//                                     <div className="flex gap-1 text-xs">
//                                         <span className="bg-purple-100 px-1 rounded">#{track.detectionCount}</span>
//                                         <span className="bg-orange-100 px-1 rounded">{Math.round(track.confidence * 100)}%</span>
//                                     </div>
//                                 </div>
//                             ))
//                         )}
//                     </div>
//                 </div>
//             </div>

//             {/* Right Panel - Enhanced Emotion Log */}
//             <div className="bg-white w-full md:w-96 h-[500px] rounded-2xl shadow-md p-4 overflow-y-auto border border-gray-200">
//                 <div className="flex justify-between items-center mb-3">
//                     <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
//                         🧠 Emotion Log
//                         <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
//                             DeepFace AI
//                         </span>
//                     </h3>
//                     <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
//                         {emotionList.length} entries
//                     </span>
//                 </div>

//                 {emotionList.length === 0 ? (
//                     <div className="text-center mt-20 text-gray-400">
//                         <p className="text-xs mt-2">Powered by DeepFace AI:</p>
//                         <ul className="text-xs text-left mt-1 space-y-1">
//                             <li>• Face recognition & tracking</li>
//                             <li>• Emotion & gender detection</li>
//                             <li>• Multi-person analysis</li>
//                         </ul>
//                     </div>
//                 ) : (
//                     <ul className="space-y-2">
//                         {emotionList.map((e, index) => (
//                             <li
//                                 key={index}
//                                 className={`flex justify-between items-start p-3 rounded-lg transition-all ${e.emotionChanged ? 'bg-green-50 border border-green-200' :
//                                     e.isNewFace ? 'bg-yellow-50 border border-yellow-200' :
//                                         'bg-gray-50 border border-gray-200'
//                                     }`}
//                             >
//                                 <div className="flex-1">
//                                     <div className="flex justify-between items-start mb-1">
//                                         <div className="flex items-center gap-2">
//                                             <span className="font-medium capitalize text-gray-800">{e.emotion}</span>
//                                             {e.gender && e.gender !== 'unknown' && (
//                                                 <span className={`text-xs px-2 py-1 rounded ${e.gender === 'man' ? 'bg-blue-100 text-blue-800' :
//                                                     'bg-pink-100 text-pink-800'
//                                                     }`}>
//                                                     {e.gender}
//                                                 </span>
//                                             )}
//                                         </div>
//                                         <div className="flex gap-1 text-xs">
//                                             <span className="bg-blue-100 px-2 py-1 rounded">{e.confidence}</span>
//                                             {e.genderConfidence && e.genderConfidence > 0.5 && (
//                                                 <span className="bg-purple-100 px-2 py-1 rounded">{e.genderConfidence}</span>
//                                             )}
//                                         </div>
//                                     </div>
//                                     <p className="text-xs text-gray-500 mb-2">{e.timestamp}</p>
//                                     <div className="flex gap-2 flex-wrap">
//                                         <span className="text-xs bg-gray-200 px-2 py-1 rounded">{e.personName}</span>
//                                         <span className="text-xs bg-purple-200 px-2 py-1 rounded">Track: {e.track_count}</span>
//                                         {e.similarity_score && (
//                                             <span className="text-xs bg-blue-200 px-2 py-1 rounded">Similarity: {e.similarity_score}</span>
//                                         )}
//                                         {e.emotionChanged && (
//                                             <span className="text-xs bg-green-200 px-2 py-1 rounded">Emotion Changed</span>
//                                         )}
//                                         {e.isNewFace && (
//                                             <span className="text-xs bg-yellow-200 px-2 py-1 rounded">New Face</span>
//                                         )}
//                                     </div>
//                                 </div>
//                             </li>
//                         ))}
//                     </ul>
//                 )}
//             </div>
//         </div>
//     );
// }
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
