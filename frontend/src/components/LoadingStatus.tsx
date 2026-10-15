import React from "react";
import { Loader2 } from "lucide-react";

interface LoadingStatusProps {
  message: string;
}

export const LoadingStatus: React.FC<LoadingStatusProps> = ({ message }) => {
  return (
    <div className="chat-status-line" role="status" aria-live="polite">
      <Loader2 size={14} className="spin" />
      <span>{message}</span>
    </div>
  );
};
