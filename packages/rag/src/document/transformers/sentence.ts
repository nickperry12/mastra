import type { ChunkOptions } from '../types';
import { TextTransformer } from './text';

export interface SentenceChunkOptions extends ChunkOptions {
  maxSize?: number;
  minSize?: number; 
  targetSize?: number;
  sentenceEnders?: string[];
  fallbackToWords?: boolean;
  fallbackToCharacters?: boolean;
}

export class SentenceTransformer extends TextTransformer {
  protected maxSize: number;
  protected minSize: number;
  protected targetSize: number;
  protected sentenceEnders: string[];
  protected fallbackToWords: boolean;
  protected fallbackToCharacters: boolean;

  constructor(options: SentenceChunkOptions = {}) {
    const sentenceOptions: ChunkOptions = {
      overlap: 50,
      ...options
    };

    // Set size to maxSize for the parent class (fallback to default)
    const maxSize = options.maxSize ?? options.size ?? 4000;
    sentenceOptions.size = maxSize;
    
    super(sentenceOptions);

    this.maxSize = maxSize;
    this.minSize = options.minSize ?? 50;
    this.targetSize = options.targetSize ?? Math.floor(maxSize * 0.8);
    this.sentenceEnders = options.sentenceEnders ?? ['.', '!', '?'];
    this.fallbackToWords = options.fallbackToWords ?? true;
    this.fallbackToCharacters = options.fallbackToCharacters ?? true;
  }

  splitText({ text }: { text: string }): string[] {
    const sentences = this.detectSentenceBoundaries(text);
    return this.groupSentencesIntoChunks(sentences);
  }

  private detectSentenceBoundaries(text: string): string[] {
    // Create regex pattern from configurable sentence enders
    const escapedEnders = this.sentenceEnders.map(ender => 
      ender.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    );
    const pattern = `([${escapedEnders.join('')}]+)`;
    
    // Split on sentence-ending punctuation while preserving the punctuation
    const parts = text.split(new RegExp(pattern));
    const sentences: string[] = [];
    let currentSentence = '';

    for (let i = 0; i < parts.length; i += 2) {
      const textPart = parts[i] || '';
      const punctuation = parts[i + 1] || '';
      
      currentSentence += textPart + punctuation;

      // If we have punctuation, check if it's likely a sentence boundary
      if (punctuation && this.isLikelySentenceBoundary(textPart, punctuation, parts[i + 2])) {
        const sentence = currentSentence.trim();
        if (sentence.length > 0) {
          sentences.push(sentence);
        }
        currentSentence = '';
      }
    }

    // Add any remaining text as the last sentence
    if (currentSentence.trim().length > 0) {
      sentences.push(currentSentence.trim());
    }

    return sentences;
  }

  private isLikelySentenceBoundary(beforePunct: string, punctuation: string, afterPunct?: string): boolean {
    // If there's no text after, it's the end of the document
    if (!afterPunct || !afterPunct.trim()) {
      return true;
    }

    // If next text doesn't start with whitespace followed by a capital letter, probably not a sentence boundary
    if (!/^\s+[A-Z]/.test(afterPunct)) {
      return false;
    }

    // Get the word immediately before the punctuation
    const words = beforePunct.trim().split(/\s+/);
    const lastWord = words[words.length - 1] || '';

    // Apply heuristics to detect common abbreviation patterns
    if (this.looksLikeAbbreviation(lastWord, punctuation)) {
      return false;
    }

    // If we get here, it's likely a real sentence boundary
    return true;
  }

  private looksLikeAbbreviation(word: string, punctuation: string): boolean {
    // Only apply abbreviation logic for single periods
    if (punctuation !== '.') {
      return false;
    }

    // Common title abbreviations (short words, often capitalized)
    if (word.length <= 4 && /^[A-Z][a-z]*$/.test(word)) {
      return true; // Dr, Mr, Mrs, Ms, Prof, etc.
    }

    // Common country/organization abbreviations (capital letters with or without periods)
    if (/^[A-Z]{1,4}$/.test(word)) {
      return true; // US, UK, USA, etc.
    }

    // Mixed case abbreviations with internal periods already
    if (/^[A-Z][a-z]*\.[A-Z]/.test(word)) {
      return true; // e.g, i.e, etc.
    }

    // Numbers (likely decimals, times, versions)
    if (/^\d+$/.test(word)) {
      return true; // 3., 12., etc.
    }

    // Time abbreviations
    if (/^[ap]\.?m$/i.test(word)) {
      return true; // a.m, p.m, am, pm
    }

    // Single letters (often initials)
    if (/^[A-Z]$/.test(word)) {
      return true; // J., etc.
    }

    return false;
  }

  private groupSentencesIntoChunks(sentences: string[]): string[] {
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentSize = 0;

    for (const sentence of sentences) {
      const sentenceLength = this.lengthFunction(sentence);
      const separatorLength = currentChunk.length > 0 ? this.lengthFunction(' ') : 0;
      const totalLength = currentSize + sentenceLength + separatorLength;

      // Handle oversized sentences with fallback strategies
      if (sentenceLength > this.maxSize) {
        // Flush current chunk first
        if (currentChunk.length > 0) {
          chunks.push(currentChunk.join(' '));
          currentChunk = [];
          currentSize = 0;
        }

        // Apply fallback strategies for oversized sentence
        const fallbackChunks = this.handleOversizedSentence(sentence);
        chunks.push(...fallbackChunks);
        continue;
      }

      // If adding this sentence would exceed maxSize, finalize current chunk
      if (currentChunk.length > 0 && totalLength > this.maxSize) {
        chunks.push(currentChunk.join(' '));

        // Calculate overlap for next chunk
        const overlapSentences = this.calculateSentenceOverlap(currentChunk);
        currentChunk = overlapSentences;
        currentSize = this.calculateChunkSize(currentChunk);
      }

      currentChunk.push(sentence);
      currentSize += sentenceLength + separatorLength;

      // If we've reached our target size, consider finalizing the chunk
      if (currentSize >= this.targetSize) {
        chunks.push(currentChunk.join(' '));

        // Calculate overlap for next chunk
        const overlapSentences = this.calculateSentenceOverlap(currentChunk);
        currentChunk = overlapSentences;
        currentSize = this.calculateChunkSize(currentChunk);
      }
    }

    // Add the final chunk if it has content
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));
    }

    return chunks;
  }

  private handleOversizedSentence(sentence: string): string[] {
    // First fallback: split by words
    if (this.fallbackToWords) {
      const wordChunks = this.splitSentenceIntoWords(sentence);
      // If word splitting produced multiple chunks, return them
      if (wordChunks.length > 1) {
        return wordChunks;
      }
    }

    // Second fallback: split by characters (only if word splitting didn't help)
    if (this.fallbackToCharacters) {
      return this.splitSentenceIntoCharacters(sentence);
    }

    // Last resort: return the oversized sentence as-is with a warning
    console.warn(`Sentence exceeds maxSize (${this.maxSize}) and fallbacks are disabled: "${sentence.substring(0, 50)}..."`);
    return [sentence];
  }

  private splitSentenceIntoWords(sentence: string): string[] {
    const words = sentence.split(/\s+/);
    const chunks: string[] = [];
    let currentChunk = '';

    for (const word of words) {
      const testChunk = currentChunk ? currentChunk + ' ' + word : word;

      if (this.lengthFunction(testChunk) <= this.maxSize) {
        currentChunk = testChunk;
      } else {
        if (currentChunk) {
          chunks.push(currentChunk);
        }
        
        // If single word is still too long, handle with character fallback
        if (this.lengthFunction(word) > this.maxSize) {
          if (this.fallbackToCharacters) {
            chunks.push(...this.splitSentenceIntoCharacters(word));
          } else {
            chunks.push(word); // Push oversized word as-is
          }
          currentChunk = '';
        } else {
          currentChunk = word;
        }
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk);
    }

    return chunks;
  }

  private splitSentenceIntoCharacters(text: string): string[] {
    const chunks: string[] = [];
    let currentChunk = '';

    for (const char of text) {
      if (this.lengthFunction(currentChunk + char) <= this.maxSize) {
        currentChunk += char;
      } else {
        if (currentChunk) {
          chunks.push(currentChunk);
        }
        currentChunk = char;
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk);
    }

    return chunks;
  }

  private calculateSentenceOverlap(currentChunk: string[]): string[] {
    if (this.overlap === 0 || currentChunk.length === 0) {
      return [];
    }

    const overlapSentences: string[] = [];
    let overlapSize = 0;

    // Work backwards through sentences to build overlap
    for (let i = currentChunk.length - 1; i >= 0; i--) {
      const sentence = currentChunk[i];
      if (!sentence) continue;

      const sentenceLength = this.lengthFunction(sentence);
      const separatorLength = overlapSentences.length > 0 ? this.lengthFunction(' ') : 0;

      if (overlapSize + sentenceLength + separatorLength > this.overlap) {
        break;
      }

      overlapSentences.unshift(sentence);
      overlapSize += sentenceLength + separatorLength;
    }

    return overlapSentences;
  }

  private calculateChunkSize(sentences: string[]): number {
    if (!sentences || sentences.length === 0) {
      return 0;
    }

    let totalSize = 0;
    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i]!;
      totalSize += this.lengthFunction(sentence);
      
      // Add separator length for all but the last sentence
      if (i < sentences.length - 1) {
        totalSize += this.lengthFunction(' ');
      }
    }

    return totalSize;
  }
}