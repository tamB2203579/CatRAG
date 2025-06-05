import { useState } from 'react';
import { assets } from '../../assets/assets';
import './Window.css';

const Window = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const createMsgElement = (content, type) => ({
    id: Date.now(),
    type,
    content
  });

  const sendMessage = async () => {
    try {
      const formData = new FormData();
      formData.append("msg", input);

      const res = await fetch("http://127.0.0.1:8000/response", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      return data.response;
    } catch (error) {
      console.error("Error fetching response:", error);
      return "Đã xảy ra lỗi khi lấy phản hồi.";
    }
  };

  const onHandleSubmit = async (e) => {
    e.preventDefault();
    const userMessage = input.trim();
    if (!userMessage) return;

    setInput('');

    const userMsg = createMsgElement(userMessage, 'user');
    setMessages((prevMessages) => [...prevMessages, userMsg]);

    const loadingMsg = createMsgElement("Vui lòng chờ trong giây lát...", "bot");
    setMessages((prevMessages) => [...prevMessages, loadingMsg]);

    try {
      const botResponse = await sendMessage();
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        updatedMessages[updatedMessages.length - 1] = createMsgElement(botResponse, "bot");
        return updatedMessages;
      });
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  return (
    <div className='main'>
      <div className="nav">
        <p>REBot</p>
      </div>
      <div className="main-container">
        <div className="chats-container">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.type}-message ${
                msg.type === "bot" ? "loading" : ""
              }`}
            >
              {msg.type === "bot" ? (
                <>
                  <img src={assets.bot_avatar} alt="" className="avatar" />
                  <p className="message-text">{msg.content || "Loading..."}</p>
                </>
              ) : (
                <p className="message-text">{msg.content}</p>
              )}
            </div>
          ))}
        </div>

        <div className="prompt-container">
          <div className='prompt-wrapper'>
            <div className='prompt-search'>
              <input 
                className='prompt-input' 
                type="text" 
                value={input} 
                onChange={(e) => setInput(e.target.value)}
                placeholder='Message REBot' 
                required
              />
              <div className='prompt-actions'>
                <button 
                  id='send-btn' 
                  className="material-symbols-outlined" 
                  onClick={onHandleSubmit}>send</button>
              </div>
            </div>
            <button id='theme-toggle-btn' className="material-symbols-outlined">light_mode</button>
            <button id='delete-btn' className="material-symbols-outlined">delete</button>
          </div>

          <p className='bottom-info'>REBot can make mistakes. Check important info.</p>
        </div>
      </div>
    </div>
  );
};

export default Window;
