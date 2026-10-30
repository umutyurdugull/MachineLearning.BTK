comment1 = "BTK akademi çok zengin"
comment2 = "BTK akademi çok faydalı"

# We need to create a pool of words with the words in each sentences.

pool = (comment1 + " " + comment2).lower().split()
vocab = sorted(list(set(pool))) # Alphabetic sorted list

print(vocab)


# Converting sentence to vector

def textToBag(text,vocab):
    vector = []
    words = text.lower().split()
    for word in vocab:
        if word in words:
            vector.append(1)
        else:
            vector.append(0)
    return vector


vector1 = textToBag(comment1,vocab)
vector2 = textToBag(comment2,vocab)

print(vector1)
print(vector2)