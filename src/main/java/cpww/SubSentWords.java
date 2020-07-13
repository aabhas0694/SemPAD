package cpww;

import edu.stanford.nlp.ling.IndexedWord;
import java.io.Serializable;
import java.util.Map;
import java.util.regex.Matcher;

import static cpww.utils.Util.pattern;
import static cpww.utils.Util.returnLowercaseLemma;

public class SubSentWords implements Comparable, Serializable {
    private String word;
    private String lemma;
    private String encoding;
    private int index;

    public SubSentWords(IndexedWord iw, String encode, boolean isEntity, Map<String, String> entityDict) {
        setWordAndLemma(iw, entityDict, isEntity);
        this.encoding = encode;
        this.index = iw.index();
    }

    public SubSentWords(String word, String lemma, String encoding, int index) {
        this.word = word;
        this.lemma = lemma;
        this.encoding = encoding;
        this.index = index;
    }

    public SubSentWords(SubSentWords subSentWords) {
        this.word = subSentWords.getOriginalWord();
        this.lemma = subSentWords.getLemma();
        this.encoding = subSentWords.getEncoding();
        this.index = subSentWords.getIndex();
    }

    public String getOriginalWord() {
        return this.word;
    }

    public String getLemma() {
        return this.lemma;
    }

    public int getIndex() {
        return this.index;
    }

    public String getEncoding() {
        return this.encoding;
    }

    public String getTrimmedEncoding() {
        return this.encoding.split("_")[0];
    }

    public void setWord(String word) {
        this.word = word;
    }

    private void setWordAndLemma(IndexedWord entity, Map<String, String> entityDict, boolean isEntity) {
        String tok1 = entity.word().trim();
        this.lemma = tok1;
        this.word = tok1;
        Matcher matcher = pattern.matcher(tok1);
        boolean patternFound = false;
        String match = null;
        while (matcher.find()) {
            match = matcher.group();
            patternFound = true;
            this.word = this.word.replace(match, entityDict.getOrDefault(match, match));
            this.lemma = isEntity ? this.lemma.replace(match, match.replaceAll("([A-Z]+)(\\d)+","$1")) : this.word;
        }
        if (!isEntity) {
            this.lemma = patternFound ? (entityDict.containsKey(match) ? this.lemma.toLowerCase() : this.lemma) : returnLowercaseLemma(entity);
        }
    }

    public void setEncoding(String encode) {
        this.encoding = encode;
    }

    @Override
    public int compareTo(Object o) {
        Integer i1 = this.index;
        Integer i2 = ((SubSentWords) o).index;
        return i1.compareTo(i2);
    }
}
