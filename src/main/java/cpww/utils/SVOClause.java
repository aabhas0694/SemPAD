package cpww.utils;

import edu.stanford.nlp.ling.IndexedWord;

public class SVOClause {
    private IndexedWord subject;
    private IndexedWord verb;
    private IndexedWord object;
    private IndexedWord conjunct;

    public SVOClause (IndexedWord verb) {
        this.verb = verb;
    }

    public void setSubject(IndexedWord subj) {
        this.subject = subj;
    }

    public void setObject(IndexedWord object) {
        this.object = object;
    }

    public void setConjunct(IndexedWord conj) {
        this.conjunct = conj;
    }

    public IndexedWord getConjunct() {
        return this.conjunct;
    }

    public IndexedWord getSubject() {
        return this.subject;
    }

    public IndexedWord getObject() {
        return this.object;
    }

    public boolean foundSubject() {
        return subject != null;
    }

    public boolean foundObject() {
        return object != null;
    }

    public boolean foundConjunt() {
        return conjunct != null;
    }

    public boolean isCoordClause() {
        return foundSubject() && foundConjunt();
    }
}
