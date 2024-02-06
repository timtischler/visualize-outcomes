import { ChatWindow } from "@/components/ChatWindow";

export default function AgentsPage() {
  const InfoCard = (
    <div className="p-4 md:p-8 rounded bg-[#25252d] w-full max-h-[85%] overflow-hidden">
      <h1 className="text-3xl md:text-4xl mb-4">
        Welformed Outcome Agent
      </h1>
      <ul>
        <li className="text-l">
          🤝
          <span className="ml-2">
            I am an agent that will help you create realistic achievable goals.  Just tell me what you want and we&apos;ll get started. 
          </span>
        </li>
      </ul>
    </div>
  );
  return (
    <ChatWindow
      endpoint="api/chat/welformed_outcome"
      emptyStateComponent={InfoCard}
      placeholder="Hello! I'm a conversational agent trained to help you create achievable goals!"
      titleText="Welformed Outcome"
      emoji="🦜"
    ></ChatWindow>
  );
}

