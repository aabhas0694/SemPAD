package sempad;

import lombok.Getter;

import java.util.*;

import static sempad.utils.Util.articles;
import static sempad.utils.Util.patternRegexFilter;

@Getter
public class PatternCandidate {
    private String[] candidate;
    private MetaPattern metaPattern;
    private boolean isMetaPattern = true;
    private Integer frequency;

    PatternCandidate(String pattern, String[] nerTypes, List<String> stopWords, int frequency) {
        this.candidate = Arrays.stream(pattern.split(" "))
                .filter(word -> !word.matches(patternRegexFilter))
                .toArray(String[]::new);
        this.setMetaPattern(stopWords, nerTypes);
        this.frequency = frequency;
    }

    private void setMetaPattern(List<String> stopWords, String[] nerTypes) {
        String[] splitPattern = this.candidate;
        Set<String> entityContainingWords = new HashSet<>();
        List<Integer> entityIndex = new ArrayList<>();

        int foundStart = -1, foundEnd = -1;
        for (int i = 0; i < splitPattern.length; i++) {
            String pat = splitPattern[i];
            for (String ner : nerTypes) {
                if (pat.contains(ner)) {
                    entityContainingWords.add(pat);
                    entityIndex.add(i);
                    break;
                }
            }
            if (!stopWords.contains(pat)) {
                if (foundStart == -1) foundStart = i;
                foundEnd = i;
            }
        }

        if (isEntityContinuous(entityIndex)) {
            this.isMetaPattern = false;
            this.metaPattern = null;
            return;
        }

        if (foundStart >= foundEnd || Arrays.asList(splitPattern).subList(foundStart, foundEnd + 1).stream()
                .noneMatch(s -> (s.contains("-") || !entityContainingWords.contains(s)) && !Arrays.asList(articles).contains(s))) {
            this.isMetaPattern = false;
            this.metaPattern = null;
            return;
        }
        this.metaPattern = new MetaPattern(Arrays.copyOfRange(splitPattern, foundStart, foundEnd + 1),
                entityContainingWords, entityIndex);

        if (this.checkConjunction(this.metaPattern.getSplitPattern())) {
            this.isMetaPattern = false;
        }
    }

    /**
     Returns true if there exist conjunctions in a pattern
     */
    private boolean checkConjunction(String[] sPattern) {
        List<String> splitPattern = Arrays.asList(sPattern);
        return ((splitPattern.contains("and") || splitPattern.contains("or"))  && !splitPattern.contains("between")) ||
                splitPattern.contains("but") || splitPattern.contains("nor");
    }

    /**
     * Checks if all entities are not placed one after the other at consecutive positions and length of pattern without hyphens is not 2.
     * @return If so, returns true, else false.
     */
    private boolean isEntityContinuous(List<Integer> entityIndex) {
        if (entityIndex.size() == 0) return true;
        else if (entityIndex.size() == 1) return false;
        String[] splitPattern = this.candidate;
        for (int i = 0; i < entityIndex.size() - 1; i++) {
            int i1 = entityIndex.get(i), i2 = entityIndex.get(i + 1);
            if (i1 + 1 == i2 && splitPattern[i1].equals(splitPattern[i2])) {
                return true;
            }
        }
        return false;
    }

    public boolean isMetaPattern() {
        return this.isMetaPattern;
    }
}
