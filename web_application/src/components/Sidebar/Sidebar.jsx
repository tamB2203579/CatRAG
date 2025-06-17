import { assets } from '../../assets/assets';
import { useEffect, useState } from 'react';
import WebFont from 'webfontloader';
import './Sidebar.css'

const Sidebar = ({ isOpen, onToggle }) => {
  const [recentChats, setRecentChats] = useState([
    'Testing...',
    'Enrollment procedure in 2025dhdjhsfjfhfhfshkfhnew',
    'Dormitory information',
    'About CSS'
  ]);

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

  const handleNewChat = () => {
    setRecentChats(prev => [...prev, `Untitled chat ${prev.length + 1}`]);
  };
  
  return (
    <div className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      <div className="top">
        <div className='menu-container'>
          <img className="menu" style={{ opacity: 1, pointerEvents: 'auto' }} src={assets.menu_icon} alt="" onClick={onToggle}/>
        </div>
        <img className="home_logo" src={assets.logo_icon} alt="" onClick={() => {window.location.reload()}} />
        <div className="new-chat" onClick={handleNewChat}>
          <img src={assets.plus_icon} alt="" />
          <p>New Chat</p>
        </div>
        <div className="recent">
          <p className="recent-title">Recent</p>
          {recentChats.map((title, index) => (
            <div className="recent-entry" key={index}>
              <img src={assets.white_message_icon} alt="" />
              <p>{title}</p>
            </div>
          ))}
        </div>
      </div>
    </div>

    
  )
};

export default Sidebar