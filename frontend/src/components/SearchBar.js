import React, { useState } from 'react';

const SearchBar = ({ onSearch, searchMode, centered = false }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSearch(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={`${centered ? 'space-y-4' : ''} w-full max-w-3xl mx-auto`}>
      <form onSubmit={handleSubmit} className="relative shadow-lg rounded-xl">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={3}
          className="focus:ring-primary-500 focus:border-primary-500 block w-full rounded-xl border border-gray-200 shadow-sm px-5 py-4 resize-none text-base"
          placeholder={centered ? "What do you want to know..." : "Ask another question..."}
        />
        <div className="absolute bottom-3 right-3 flex items-center">
          <button
            type="submit"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-full shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            disabled={!inputValue.trim()}
          >
            Search
          </button>
        </div>
      </form>
      <p className="text-xs text-gray-500 text-center mt-2">
        Press Enter to search, Shift+Enter for new line
      </p>
    </div>
  );
};

export default SearchBar; 