import { assets } from '../../assets/assets';
import { useEffect } from 'react';
import WebFont from 'webfontloader';
import './Sidebar.css'

const Sidebar = () => {

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

  return (
    <div className="sidebar">
      <div className="top">
        <img className="menu" src={assets.menu_icon} alt="" />
        <img className="home_logo" src={assets.logo_icon} alt="" onClick={() => {window.location.reload()}} />
        <div className="new-chat">
          <img src={assets.plus_icon} alt="" />
          <p>New Chat</p>
        </div>
        <div  className='recent'>
          <p className='recent-title'>Recent</p>
          <div className='recent-entry'>
            <img src={assets.message_icon} alt="" />
            <p>Testing...</p>
          </div>
        </div>
      </div>
    </div>

    
  )
};

export default Sidebar