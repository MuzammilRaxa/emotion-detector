// lib/websocket.js or hooks/useWebSocket.js
export const getWebSocketUrl = () => {
    // For client-side
    if (typeof window !== 'undefined') {
        return process.env.NEXT_PUBLIC_WS_URL;
    }
    // For server-side (optional)
    return process.env.WS_URL || 'ws://localhost:8765';
};