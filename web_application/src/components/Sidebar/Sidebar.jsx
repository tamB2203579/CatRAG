import { useEffect, useState } from 'react';
import { doc, collection, addDoc, getDoc, getDocs, Timestamp } from 'firebase/firestore';
import { db } from '../../../firebase.config';
import WebFont from 'webfontloader';
import { assets } from '../../assets/assets';
import './Sidebar.css';

const Sidebar = ({ isOpen, onToggle, updateCurrentThread, updateChatHistory }) => {
  const [threads, setThreads] = useState([]);

  useEffect(() => {
    WebFont.load({
      google: {
        families: [
          'K2D:400,500,700&display=swap',
          'Readex Pro:400,500,700&display=swap',
        ],
      },
    });

    const fetchThreads = async () => {
      try {
        const threadsCollection = collection(db, 'threads');
        const threadsSnapshot = await getDocs(threadsCollection);
        const threadsList = threadsSnapshot.docs.map((doc) => ({
          id: doc.id,
          title: doc.data().title || 'Cuộc trò chuyện mới',
          timestamp: doc.data().timestamp,
        }));
        setThreads(threadsList.sort((a, b) => b.timestamp.seconds - a.timestamp.seconds));
      } catch (error) {
        console.error('Lỗi tải danh sách thread:', error);
      }
    };

    fetchThreads();
  }, []);

  const createThread = async () => {
    try {
      const threadRef = await addDoc(collection(db, 'threads'), {
        title: 'Cuộc trò chuyện mới',
        contents: [],
        timestamp: Timestamp.now(),
      });

      const newThread = {
        id: threadRef.id,
        title: 'Cuộc trò chuyện mới',
        timestamp: Timestamp.now(),
      };
      setThreads((prev) => [newThread, ...prev.sort((a, b) => b.timestamp.seconds - a.timestamp.seconds)]);

      handleThreadClick(threadRef.id);
    } catch (error) {
      console.error('Lỗi tạo thread:', error);
    }
  };

  const handleThreadClick = async (threadId) => {
    try {
      console.log(`Chuyển sang thread ${threadId}`);
      updateCurrentThread(threadId);

      const threadRef = doc(db, 'threads', threadId);
      const threadSnap = await getDoc(threadRef);

      if (threadSnap.exists()) {
        const threadData = threadSnap.data();
        updateChatHistory(threadData.contents || []);
      } else {
        console.warn('Không tìm thấy thread');
        updateChatHistory([]);
      }
    } catch (error) {
      console.error('Lỗi tải thread:', error);
      updateChatHistory([]);
    }
  };

  return (
    <div className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      <div className="top">
        <div className="menu-container">
          <img
            className="menu"
            style={{ opacity: 1, pointerEvents: 'auto' }}
            src={isOpen ? assets.white_menu_icon : assets.menu_icon}
            alt=""
            onClick={onToggle}
          />
        </div>

        <img
          className="home_logo"
          src={assets.logo_icon}
          alt=""
          onClick={() => window.location.replace('/landing.html')}
        />

        <div className="new-chat" onClick={createThread}>
          <img src={assets.plus_icon} alt="" />
          <p>Cuộc trò chuyện mới</p>
        </div>

        <div className="recent">
          <p className="recent-title">Gần đây</p>
          <div className="recent-list">
            {threads.map((thread) => (
              <div
                className="recent-entry"
                key={thread.id}
                onClick={() => handleThreadClick(thread.id)}
              >
                <img src={assets.white_message_icon} alt="" />
                <p>{thread.title}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div>
        <img
          className="home-btn"
          src={isOpen ? assets.white_home_icon : assets.home_icon}
          alt=""
          onClick={() => window.location.replace('/landing.html')}
        />
      </div>
    </div>
  );
};

export default Sidebar;