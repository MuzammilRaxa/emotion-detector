import EmotionTracker from "@/components/EmotionTracker";
import EmotionTrackerPy from "@/components/EmotionTrackerPython";
import EnhancedEmotionTracker from "@/components/EnhancedEmotionTracker";
import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen">
      <main className="flex min-h-screen w-full">
        {/* <EmotionTracker /> */}
        <EmotionTrackerPy />
      </main>
    </div>
  );
}
