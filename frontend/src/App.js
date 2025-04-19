import React, { useState, useEffect, useRef } from 'react';
import SearchBar from './components/SearchBar';
import ResultDisplay from './components/ResultDisplay';
import { mockData } from './mockData';

function App() {
  const [question, setQuestion] = useState('');
  const [searchMode, setSearchMode] = useState('bm25'); // default: bm25, options: bm25, glove, colbert
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const [showAnimation, setShowAnimation] = useState(false);
  const mainContentRef = useRef(null);

  // Scroll to bottom when results change
  useEffect(() => {
    if (mainContentRef.current && (results || showAnimation)) {
      mainContentRef.current.scrollTo({
        top: mainContentRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [results, searchHistory, showAnimation]);

  const handleSearch = async (inputQuestion) => {
    setLoading(true);
    setQuestion(inputQuestion);
    setShowAnimation(true);
    
    // Save current question and results to history before getting new results
    if (results) {
      setSearchHistory([...searchHistory, {
        question: question,
        results: results,
        searchMode: searchMode
      }]);
    }
    
    // Simulate API call with setTimeout
    setTimeout(() => {
      const data = mockData[searchMode];
      setResults(data);
      setLoading(false);
    }, 1000);

    // For future real API implementation
    // try {
    //   const response = await fetch('/api/search', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({
    //       question: inputQuestion,
    //       mode: searchMode
    //     }),
    //   });
    //   const data = await response.json();
    //   setResults(data);
    // } catch (error) {
    //   console.error('Error fetching results:', error);
    // } finally {
    //   setLoading(false);
    // }
  };

  const resetSearch = () => {
    setQuestion('');
    setResults(null);
    setSearchHistory([]);
    setShowAnimation(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-['Inter',system-ui,sans-serif]">
      <header className="bg-white shadow-sm flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">KBQA</h1>
          <div className="flex items-center space-x-3">
            <div className="text-sm text-gray-600">Search Mode:</div>
            <div className="inline-flex rounded-md shadow-sm" role="group">
              <button
                type="button"
                onClick={() => setSearchMode('bm25')}
                className={`px-3 py-1.5 text-sm font-medium rounded-l-md ${
                  searchMode === 'bm25'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border border-gray-300`}
              >
                BM25
              </button>
              <button
                type="button"
                onClick={() => setSearchMode('glove')}
                className={`px-3 py-1.5 text-sm font-medium ${
                  searchMode === 'glove'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border-t border-b border-gray-300`}
              >
                GloVe
              </button>
              <button
                type="button"
                onClick={() => setSearchMode('colbert')}
                className={`px-3 py-1.5 text-sm font-medium rounded-r-md ${
                  searchMode === 'colbert'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border border-gray-300`}
              >
                ColBERT
              </button>
            </div>
            {(results || searchHistory.length > 0) && (
              <button
                onClick={resetSearch}
                className="ml-2 text-sm text-gray-600 hover:text-gray-900 font-medium flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Clear
              </button>
            )}
          </div>
        </div>
      </header>
      
      <main className="flex-1 overflow-auto" ref={mainContentRef}>
        <div className="max-w-5xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          {/* Initial centered search when no results */}
          {!results && !showAnimation && searchHistory.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[70vh]">
              <div className="w-full max-w-2xl">
                <h2 className="text-2xl font-bold text-center text-gray-900 mb-6 tracking-tight">
                  Knowledge Base Question Answering
                </h2>
                <SearchBar 
                  onSearch={handleSearch} 
                  searchMode={searchMode}
                  centered={true}
                />
              </div>
            </div>
          )}
          
          {/* Search history display */}
          {searchHistory.length > 0 && (
            <div className="space-y-8 mb-8">
              {searchHistory.map((item, index) => (
                <div key={index} className="space-y-6 animate-fadeIn">
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-3">
                      <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold">
                        Q
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow-sm inline-block max-w-[85%]">
                      <p className="text-gray-800">{item.question}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-3">
                      <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center text-primary-800 font-bold">
                        A
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow-sm inline-block max-w-[85%]">
                      <ResultDisplay results={item.results} question={item.question} isHistoryItem={true} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Current search and results */}
          {(results || (loading && showAnimation)) && (
            <div className="space-y-6 animate-slideUp">
              <div className="flex items-start">
                <div className="flex-shrink-0 mr-3">
                  <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-800 font-bold">
                    Q
                  </div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm inline-block max-w-[85%]">
                  <p className="text-gray-800">{question}</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="flex-shrink-0 mr-3">
                  <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center text-primary-800 font-bold">
                    A
                  </div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm inline-block max-w-[85%]">
                  {loading ? (
                    <div className="flex justify-center items-center py-6">
                      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-600"></div>
                    </div>
                  ) : (
                    <ResultDisplay results={results} question={question} />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      {/* Fixed search bar at bottom when results are shown */}
      {(results || searchHistory.length > 0 || showAnimation) && (
        <div className=" border-gray-200 py-6 flex-shrink-0">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <SearchBar 
              onSearch={handleSearch} 
              searchMode={searchMode}
              centered={false}
            />
          </div>
        </div>
      )}
      
      <footer className="bg-white border-t border-gray-200 py-3 text-center text-sm text-gray-500 flex-shrink-0">
        Knowledge Base Question Answering System - COMP5423 Project
      </footer>
    </div>
  );
}

export default App; 