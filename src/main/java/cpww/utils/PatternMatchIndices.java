package cpww.utils;

import java.util.List;

public class PatternMatchIndices {
    private List<Integer> elementIndices;
    private List<Integer> entityIndices;

    PatternMatchIndices(List<Integer> elementIndices, List<Integer> entityIndices) {
        this.elementIndices = elementIndices;
        this.entityIndices = entityIndices;
    }

    public List<Integer> getElementIndices() {
        return elementIndices;
    }

    public List<Integer> getEntityIndices() {
        return entityIndices;
    }
}
