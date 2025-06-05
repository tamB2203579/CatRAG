import { assets } from '../../assets/assets';
import './Sidebar.css'

const Sidebar = () => {

  return (
    <div className="sidebar">
      <div className="top">
        <div className="new-chat">
          <img src={assets.plus_icon} alt="" />
          <p>New Chat</p>
        </div>
        {/* <div className="recent">
          <p className="recent-title">Recent</p>
        </div> */}
      </div>
      <div className="bottom">
        <div className="bottom-item">
          <button id='help-btn' className="material-symbols-outlined">help</button>
          <p>Help</p>
        </div>

        <div className="bottom-item">
          <button id='history-btn' className="material-symbols-outlined">history</button>
          <p>Activity</p>
        </div>

        <div className="bottom-item">
          <button id='settings-btn' className="material-symbols-outlined">settings</button>
          <p>Setting</p>
        </div>

        <hr className="divider" />

        <div className="bottom-item sign-out">
        <button id='logout-btn' className="material-symbols-outlined">logout</button>
          <p>Sign Out</p>
        </div>
      </div>
    </div>

    
  )
};

export default Sidebar