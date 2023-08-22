package dl4jtest;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Word2VecTest {

	public static void main(String[] args) {

		SentenceIterator sIterator = new LineSentenceIterator(new File("data/movie_titles.txt"));
		
		TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		
		Word2Vec w2Vec = new Word2Vec.Builder()
				//.minWordFrequency(2)
				.iterations(1)
				.layerSize(1500)
				.seed(42)
				.windowSize(5)
				.iterate(sIterator)
				.tokenizerFactory(tokenizerFactory)
				.build();

		w2Vec.fit();
		System.out.println("Fitting Word2Vec model...");

		WordVectorSerializer.writeWord2VecModel(w2Vec, "data/word2Vec_output.txt");
		
		// System.out.println("closest words to \"Alien\":");
		// Collection<String> wordsNearest = w2Vec.wordsNearest("alien", 10);
		// System.out.println(wordsNearest);
		
		System.out.println("Enter a word to see similar words:");
		
		Scanner inputScanner = new Scanner(System.in);
		String inputStr = inputScanner.nextLine();

		System.out.println();

		while (!inputStr.equals("quit")) {
			
			Collection<String> wordsNearest = w2Vec.wordsNearest(inputStr, 10);
			System.out.println(wordsNearest);
			System.out.println();

			System.out.println("Enter another word to see similar words:");

			inputStr = inputScanner.nextLine();

			System.out.println();
		}
	}

}
