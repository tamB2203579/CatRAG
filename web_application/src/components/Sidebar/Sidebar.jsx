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
          'K2D:vietnamese',
          'Readex Pro:vietnamese'
        ]
        }
      });
  }, []);

  const handleNewChat = () => {
    setRecentChats(prev => [`Untitled chat ${prev.length + 1}`, ...prev]);
  };
  
  return (
    <div className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      <div className="top">
        <div className='menu-container'>
          <img className="menu" style={{ opacity: 1, pointerEvents: 'auto' }} src={isOpen ? assets.white_menu_icon : assets.menu_icon} alt="" onClick={onToggle}/>
        </div>
        <img className="home_logo" src={assets.logo_icon} alt="" onClick={() => {window.location.replace("/landing.html");}} />
        <div className="new-chat" onClick={handleNewChat}>
          <img src={assets.plus_icon} alt="" />
          <p>New Chat</p>
        </div>
        <div className="recent">
          <p className="recent-title">Recent</p>
          <div className="recent-list">
            {recentChats.map((title, index) => (
              <div className="recent-entry" key={index}>
                <img src={assets.white_message_icon} alt="" />
                <p>{title}</p>
              </div>
            ))}
           </div>
        </div>
      </div>
      <div>
        <img className="home-btn" src={isOpen ? assets.white_home_icon : assets.home_icon} alt="" onClick={() => {window.location.replace("/landing.html");}} />
      </div>
    </div>
  )
};

export default Sidebar