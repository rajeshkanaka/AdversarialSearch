
import React, { useState } from 'react';
import { SectionId } from './types';
import Sidebar from './components/Sidebar';
import IntroductionSection from './components/sections/IntroductionSection';
import MinimaxSection from './components/sections/MinimaxSection';
import AlphaBetaSection from './components/sections/AlphaBetaSection';
import MultiplayerGamesSection from './components/sections/MultiplayerGamesSection';
import ImperfectDecisionsSection from './components/sections/ImperfectDecisionsSection';
import StochasticGamesSection from './components/sections/StochasticGamesSection';
import StateOfTheArtSection from './components/sections/StateOfTheArtSection';


const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<SectionId>(SectionId.Introduction);

  const renderSection = () => {
    switch (activeSection) {
      case SectionId.Introduction:
        return <IntroductionSection />;
      case SectionId.Minimax:
        return <MinimaxSection />;
      case SectionId.AlphaBeta:
        return <AlphaBetaSection />;
      case SectionId.MultiplayerGames:
        return <MultiplayerGamesSection />;
      case SectionId.ImperfectDecisions:
        return <ImperfectDecisionsSection />;
      case SectionId.StochasticGames:
        return <StochasticGamesSection />;
      case SectionId.StateOfTheArt:
        return <StateOfTheArtSection />;
      default:
        return <IntroductionSection />;
    }
  };

  return (
    <div className="flex min-h-screen font-sans">
      <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
      <main className="flex-1 p-6 sm:p-8 md:p-12 bg-slate-100 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
          {renderSection()}
        </div>
      </main>
    </div>
  );
};

export default App;
