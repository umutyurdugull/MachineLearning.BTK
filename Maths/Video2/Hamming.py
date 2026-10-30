def textToBag(text,vocab):
    vector = []
    words = text.lower().split()
    for word in vocab:
        if word in words:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def Hamming(vector1,vector2):
    length = len(vector1)
    count = 0
    for i in range(length):
        a_i = vector1[i]
        b_i = vector2[i]
        if(a_i != b_i):
            count = count + 1

    return count





comment1 = "BTK Akademi yapay zeka müfredatı çok zengin"
comment2 = "Yapay zeka öğrenmeye başlayanlar için BTK Akademi çok faydalı bir kaynak"

pool = (comment1 + " " + comment2).lower().split()
vocab = sorted(list(set(pool)))

# It'll return only 1's and 0's.

vector1 = textToBag(comment1,vocab)
vector2 = textToBag(comment2,vocab)

print(Hamming(vector1,vector2))
