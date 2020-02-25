package cpww;

import edu.stanford.nlp.ling.IndexedWord;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static cpww.Util.returnLowercaseLemma;

public class SubSentWords implements Comparable, Serializable {
    private String word;
    private String lemma;
    private String encoding;
    private int index;

    public SubSentWords(IndexedWord iw, String encode, boolean isEntity) {
        this.word = iw.value();
        this.lemma = isEntity ? this.preprocess_rollup(this.getOriginalWord()) : returnLowercaseLemma(iw);
        this.encoding = encode;
        this.index = iw.index();
    }

    public SubSentWords(String word, String lemma, String encoding, int index) {
        this.word = word;
        this.lemma = lemma;
        this.encoding = encoding;
        this.index = index;
    }

    private String preprocess_rollup(String word) {
        List<String> ans = new ArrayList<>();
        String tok1 = word.replaceAll("\n","");
        Pattern pattern = Pattern.compile("[A-Z]+[\\d]+");
        Matcher matcher = pattern.matcher(tok1);
        while (matcher.find()) {
            String match = matcher.group();
            tok1 = tok1.replace(match, match.replaceAll("([A-Z]+)(\\d)+","$1"));
        }
        return tok1;
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

    @Override
    public int compareTo(Object o) {
        Integer i1 = this.index;
        Integer i2 = ((SubSentWords) o).index;
        return i1.compareTo(i2);
    }
}
