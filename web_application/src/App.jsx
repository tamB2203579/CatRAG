import React, { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar/Sidebar'
import Window from './components/Window/Window'
import './index.css'
import WebFont from 'webfontloader'

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

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

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };

  return (
    <div>
      <Sidebar isOpen={isSidebarOpen} onToggle={toggleSidebar} />
      <Window isSidebarOpen={isSidebarOpen} />
    </div>
  );
};

export default App