import { useEffect, useRef, useState } from 'react';
import { assets } from '../../assets/assets';
import WebFont from 'webfontloader';
import './Window.css';


const Window = ({ isSidebarOpen }) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  useEffect(() => {
    WebFont.load({
      google: {
        families: [
        'K2D:400,500,700&display=swap',
        'Readex Pro:400,500,700&display=swap'
      ]
      }
    });
  }, []);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


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

  const typeText = (text) => {
    let index = 0;
    const typingSpeed = 20;

    const interval = setInterval(() => {
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastMessage = updatedMessages[updatedMessages.length - 1];

        if (lastMessage && lastMessage.type === 'bot') {
          updatedMessages[updatedMessages.length - 1] = {
            ...lastMessage,
            content: text.slice(0, index + 1),
          };
        }

        return updatedMessages;
      });

      index++;

      if (index >= text.length) {
        clearInterval(interval);
      }
    }, typingSpeed);
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
      // setMessages((prevMessages) => {
      //   const updatedMessages = [...prevMessages];
      //   updatedMessages[updatedMessages.length - 1] = createMsgElement(botResponse, "bot");
      //   return updatedMessages;
      // });
      typeText(botResponse);
    } catch (error) {
      console.error("Error sending message:", error);
    }

    if (!hasSubmitted) {
      setHasSubmitted(true);
    }
  };

  return (
    <div className={`main ${isSidebarOpen ? 'sidebar-open' : 'sidebar-collapsed'}`}>
      <div className="nav">
        <p style={{fontFamily: 'K2D, sans serif'}}>REBot</p>
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
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className={`prompt-container ${hasSubmitted ? 'at-bottom' : 'centered'}`}>
          <div className='prompt-wrapper'>
            <div className='prompt-search'>
              <input 
                className='prompt-input' 
                type="text" 
                value={input} 
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      onHandleSubmit(e);
                    }
                  }
                }
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
            <button id='delete-btn' className="material-symbols-outlined" onClick={() => {setInput('')}}>delete</button>
          </div>

          <p className='bottom-info'>REBot can make mistakes. Check important info.</p>
        </div>
    </div>
  );
};

export default Window;