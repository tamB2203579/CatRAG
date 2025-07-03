import { useEffect, useRef, useState } from 'react';
import { assets } from '../../assets/assets';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { db } from '../../../firebase.config';
import WebFont from 'webfontloader';
import Sidebar from '../Sidebar/Sidebar';
import { marked } from "marked";
import './Window.css';

const Window = ({ isOpen, onToggle }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [currentThread, setCurrentThread] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    WebFont.load({
      google: {
        families: [
          'K2D:400,500,700&display=swap',
          'Readex Pro:400,500,700&display=swap',
        ],
      },
    });
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const loadThreadHistory = async () => {
      if (currentThread) {
        try {
          const threadRef = doc(db, 'threads', currentThread);
          const threadSnap = await getDoc(threadRef);
          if (threadSnap.exists()) {
            const threadData = threadSnap.data();
            setMessages(threadData.contents || []);
            setHasSubmitted(threadData.contents?.length > 0);
          } else {
            setMessages([]);
            setHasSubmitted(false);
          }
        } catch (error) {
          console.error('Lỗi tải lịch sử thread:', error);
          setMessages([]);
          setHasSubmitted(false);
        }
      } else {
        setMessages([]);
        setHasSubmitted(false);
      }
    };

    loadThreadHistory();
  }, [currentThread]);

  const createMsgElement = (content, type, loading = false) => ({
    id: Date.now(),
    type,
    content,
    loading,
    timestamp: new Date().toISOString(),
  });

  const updateThreadContent = async (updatedHistory) => {
    if (!currentThread) {
      console.log('Chưa chọn thread');
      return;
    }
    console.log('Đang cập nhật nội dung thread...');
    const threadRef = doc(db, 'threads', currentThread);
    try {
      await updateDoc(threadRef, {
        contents: updatedHistory,
      });
      console.log('Cập nhật nội dung thread thành công');
    } catch (e) {
      console.error('Lỗi cập nhật nội dung thread:', e);
    }
  };

  const updateCurrentThread = (threadId) => {
    setCurrentThread(threadId);
  };

  const formatMarkdownToHTML = (markdownText) => {
    return marked.parse(markdownText || '');
  };


  const updateChatHistory = (newChatHistory) => {
    setMessages(newChatHistory);
    setHasSubmitted(newChatHistory.length > 0);
  };

  const onHandleSubmit = async (e) => {
  e.preventDefault();
  const userMessage = input.trim();
  if (!userMessage) return;

  if (!currentThread) {
    alert('Vui lòng chọn hoặc tạo cuộc trò chuyện trước.');
    return;
  }

  setInput('');
  setHasSubmitted(true);

  const userMsg = createMsgElement(userMessage, 'user');
  setMessages((prevMessages) => [...prevMessages, userMsg]);

  try {
    setIsLoading(true);

    const loadingMsg = createMsgElement('', 'bot', true);
    setMessages((prevMessages) => [...prevMessages, loadingMsg]);

    const res = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: userMessage,
        session_id: currentThread,
        response: ''
      }),
    });

    const data = await res.json();
    const botResponse = data?.data?.response;

    const htmlContent = formatMarkdownToHTML(botResponse);

    setIsLoading(false);

    if (botResponse) {
      setMessages((prevMessages) => {
        const filtered = prevMessages.filter(
          (msg) => !msg.loading && msg.content.trim() !== ''
        );
        const botMsg = createMsgElement(htmlContent, 'bot');
        const updated = [...filtered, botMsg];

        updateThreadContent(updated);
        return updated;
      });
    }
  } catch (error) {
    console.error('Lỗi:', error);
    setIsLoading(false);
    setMessages((prevMessages) =>
      prevMessages.filter((msg) => !msg.loading)
    );
  }
};


  return (
    <div>
      <Sidebar
        isOpen={isOpen}
        onToggle={onToggle}
        updateCurrentThread={updateCurrentThread}
        updateChatHistory={updateChatHistory}
      />
      <div className={`main ${isOpen ? 'sidebar-open' : 'sidebar-collapsed'}`}>
        <div className="nav">
          <p style={{ fontFamily: 'K2D, sans-serif' }}>REBot</p>
        </div>
        <div className="main-container">
          <div className="chats-container">
            {messages.map((msg, index) => (
              <div
                key={msg.id || index}
                className={`message ${msg.type}-message ${
                  msg.loading ? 'loading' : ''
                }`}
              >
                {msg.type === 'bot' ? (
                  <>
                    <img src={assets.bot_avatar} alt="" className="avatar" />
                    <div
                      className="message-text"
                      dangerouslySetInnerHTML={{
                        __html: msg.loading ? 'Đang tải...' : msg.content,
                      }}
                    ></div>
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
          <div className="prompt-wrapper">
            <div className="prompt-search">
              <input
                className="prompt-input"
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    onHandleSubmit(e);
                  }
                }}
                placeholder="Nhắn tin cho REBot"
                required
              />
              <div className="prompt-actions">
                <button
                  id="send-btn"
                  className="material-symbols-outlined"
                  onClick={onHandleSubmit}
                >
                  send
                </button>
              </div>
            </div>
            <button id="theme-toggle-btn" className="material-symbols-outlined">
              light_mode
            </button>
            <button
              id="delete-btn"
              className="material-symbols-outlined"
              onClick={() => setInput('')}
            >
              delete
            </button>
          </div>
          <p className="bottom-info">REBot có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng.</p>
        </div>
      </div>
    </div>
  );
};

export default Window;