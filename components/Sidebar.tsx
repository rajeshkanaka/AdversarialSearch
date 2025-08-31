
import React, { useState } from 'react';
import { SectionId } from '../types';
import { SECTIONS } from '../constants';

interface SidebarProps {
  activeSection: SectionId;
  setActiveSection: (section: SectionId) => void;
}

const NavItem: React.FC<{
  id: SectionId;
  title: string;
  isActive: boolean;
  onClick: () => void;
}> = ({ id, title, isActive, onClick }) => (
  <li>
    <button
      onClick={onClick}
      className={`w-full text-left pl-8 pr-4 py-2 text-sm rounded-md transition-colors duration-200 ${
        isActive
          ? 'bg-indigo-100 text-indigo-700 font-semibold'
          : 'text-slate-600 hover:bg-slate-100'
      }`}
    >
      {title}
    </button>
  </li>
);

const Sidebar: React.FC<SidebarProps> = ({ activeSection, setActiveSection }) => {
  const [openSections, setOpenSections] = useState<string[]>(['group-intro', 'group-algorithms']);

  const toggleSection = (groupId: string) => {
    setOpenSections(prev => 
      prev.includes(groupId) ? prev.filter(id => id !== groupId) : [...prev, groupId]
    );
  };

  return (
    <nav className="w-72 bg-white shadow-lg p-4 flex-shrink-0 flex flex-col">
      <div className="flex items-center mb-6 pl-2">
         <div className="bg-indigo-600 p-2 rounded-lg mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
         </div>
        <h1 className="text-xl font-bold text-slate-800">Adversarial Search</h1>
      </div>
      <ul className="space-y-4 flex-1">
        {SECTIONS.map((group) => (
          <li key={group.id}>
            <button 
              onClick={() => toggleSection(group.id)}
              className="w-full flex items-center justify-between text-left px-3 py-2 rounded-md hover:bg-slate-100"
            >
              <span className="text-sm font-semibold text-slate-500 uppercase tracking-wider">{group.title}</span>
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 text-slate-500 transition-transform duration-200 ${openSections.includes(group.id) ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
            {openSections.includes(group.id) && (
              <ul className="mt-2 space-y-1">
                {group.children.map((section) => (
                  <NavItem
                    key={section.id}
                    id={section.id}
                    title={section.title}
                    isActive={activeSection === section.id}
                    onClick={() => setActiveSection(section.id)}
                  />
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
      <div className="text-xs text-slate-400 w-full p-2">
        <p>Based on "AI: A Modern Approach" by Stuart Russell & Peter Norvig.</p>
      </div>
    </nav>
  );
};

export default Sidebar;
