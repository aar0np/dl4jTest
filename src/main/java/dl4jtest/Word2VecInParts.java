package dl4jtest;

import java.io.File;

import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Word2VecInParts {

	public static void main(String[] args) {

		SentenceIterator sIterator = new LineSentenceIterator(new File("data/movie_titles.txt"));

		TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		
		while (sIterator.hasNext()) {
			String movie = sIterator.nextSentence();
			Tokenizer tokenizer = tokenizerFactory.create(movie);

			while (tokenizer.hasMoreTokens()) {
				String token = tokenizer.nextToken();
				System.out.println(token);
			}
		}
				
	}

}
