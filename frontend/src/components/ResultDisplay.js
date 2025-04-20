import React, { useState } from 'react';

const ResultDisplay = ({ results, question, isHistoryItem = false }) => {
  const { answer, documents } = results;
  const [isExpanded, setIsExpanded] = useState(false);
  const [showReferences, setShowReferences] = useState(!isHistoryItem);

  return (
    <div className="space-y-3">
      {/* Answer Section */}
      <div className="text-gray-800">
        <p className="text-base leading-relaxed">{answer}</p>
      </div>
      
      {/* References toggle button */}
      <div className="flex items-center">
        <button 
          onClick={() => setShowReferences(!showReferences)} 
          className="text-primary-600 hover:text-primary-800 text-sm font-medium flex items-center"
        >
          <span className="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            References
          </span>
          <span className="ml-2 bg-gray-100 text-xs font-semibold px-2.5 py-0.5 rounded-full">
            {documents.length}
          </span>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className={`h-4 w-4 ml-2 transition-transform duration-200 ${showReferences ? 'rotate-180' : ''}`} 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Document References */}
      {showReferences && (
        <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-100">
          <div className="space-y-3">
            {documents.map((doc, index) => (
              <DocumentItem key={index} document={doc} isFirst={index === 0} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const DocumentItem = ({ document, isFirst }) => {
  const [expanded, setExpanded] = useState(isFirst);
  const { title, content, score, doc_id, chunk_id } = document;
  const lineThreshold = 100;
  
  // 格式化标题：根据长度和>符号判断如何显示
  const formatTitle = (title) => {
    // 如果标题不够长，直接显示完整标题
    if (title.length <= lineThreshold) {
      return title;
    }
    
    // 如果标题中包含>符号
    if (title.includes('>')) {
      const parts = title.split('>');
      
      // 如果只有一个>，则正常显示，但如果超过60字符还是需要截断
      if (parts.length <= 2) {
        return title.length > lineThreshold ? title.substring(0, lineThreshold-3) + '...' : title;
      }
      
      // 提取第一部分和最后部分
      const firstPart = parts[0].trim();
      const lastPart = parts[parts.length - 1].trim();
      
      return `${firstPart} > ... > ${lastPart}`;
    }
    
    // 如果没有>符号但标题过长，使用截断
    return title.substring(0, lineThreshold - 3) + '...';
  };
  
  // 应用标题格式化
  const displayTitle = formatTitle(title);
  
  // Truncate content for preview
  const previewContent = content.length > 150 
    ? content.substring(0, 150) + '...' 
    : content;
  
  return (
    <div className="border border-gray-200 rounded-md overflow-hidden bg-white">
      <div 
        className="flex justify-between items-center px-3 py-2 cursor-pointer hover:bg-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <div 
          className="font-medium text-gray-700 text-sm flex-1 truncate" 
          title={title}
        >
          {displayTitle}
        </div>
        <div className="flex items-center flex-shrink-0">
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 mr-2">
            {score.toFixed(2)}
          </span>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      
      {!expanded && (
        <div className="px-3 py-2 text-xs text-gray-500 border-t border-gray-100">
          {previewContent}
        </div>
      )}
      
      {expanded && (
        <div className="px-3 py-2 border-t border-gray-100">
          <div className="text-sm text-gray-700 mb-2">
            {content}
          </div>
          <div className="text-xs text-gray-500 flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded-full bg-gray-100">ID: {doc_id}</span>
            <span className="px-2 py-0.5 rounded-full bg-gray-100">Chunk: {chunk_id}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay; 