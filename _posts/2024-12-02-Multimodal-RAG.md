## Multimodal RAG (MMRAG)

Retrieval Augmented Generation and other RAG-based methods like CAG and KAG focuses solely on retrieving textual data. Textual data, however, has limitations, as many documents, especially online ones, not only have the main text but supporting figures, images, or even related videos.
What if now you could retrieve images, audios, or even videos for answering a query? The retrieval also is typically confined to one vector space, and if you want to retrieve data from multiple domains, you need to have multiple separate storages. If you could combine all different domains coincide in one space, it would be extremely efficient. 
MMRAG not only enables retrieval of multiple modalities, such as text, images, audios, and videos, but it could also unify these modalities into a single space. 

-----------------------------------------------------------------------------------

### Contrastive Learning
<img width="742" height="204" alt="image" src="https://github.com/user-attachments/assets/f8110ef0-5e5b-4258-a1f7-fddce6ee13c1" />


To conjoin two separate modalities from distinct embedding spaces, we employ what's called constrastive learning. As the image suggests, how constrastive learning works is that when a comparison is made between two elements from different modalities, we keep similar ones closer and dissimilar ones further. 
For instance, suppose you want to unify an image embedding space and an audio embedding space. Let's say that there is ImageA that depicts a car and ImageB that depicts a cat. Suppose you also have Text1 that describes Lamborghini and Text2 that says "Meow". (ImageA and Text1) and (ImageB and Text2) would be put closer to each other in a new embedding space, but any other pairings would be put further apart.
The basic idea, therefore is to place elements with high conceptual similarities closer and elements with low conceptual similarities further.
This rule is applied for all elements whether they are from the same modality.

Then now the question is "how would you define a metric for how close similar things should be and how far dissimilar things should be"?
That answer is in the loss function.

### Constrastive loss

One of the losses is [constrastive loss](https://ieeexplore.ieee.org/document/1467314 ), a very original approach proposed in 2005.
Let's say you have two different elements $$x_i$$ and $$x_j$$ from two different embedding spaces. 
For this pair, the loss is given as:

<img width="980" height="48" alt="image" src="https://github.com/user-attachments/assets/dfc858b1-0334-4584-8268-9d77c4fc5c2c" />

where
+ $$y_i \in \set{0,...,i,...,k}, k= |Y|+1 $$
+ $$f_\theta \left( x \right): X \to \mathbb{R}^d$$,
+ $$\epsilon $$ = minimum distance between samples of dissimilar classes

What this equation is saying is that when $$y_i$$ and $$y_j$$, class labels of $$x_i$$ and $$x_j$$, respectively, are equal to each other, the loss is equal to the euclidean distance between unified encodings of $$x_i$$ and $$x_j$$, which can be also written as the following.
$$||f_\theta (x_i) - f_\theta (x_j)||^2_2 = \left( \sqrt{\sum_{k=1}^{d} ( f_\theta (x_i)_k - f _\theta (x_j)_k )^2 } \right)^2$$

Notice that the right handside resembles Mean Squared Error (MSE) inside the square root. Within the same class, the closer the two embeddings are, the smaller the loss becomes.
For when they are of different classes, we check three different cases:

(letting D = $$\epsilon$$ - the euclidean distance)
+ if $$\epsilon$$ > the euclidean distance, then D>0, the loss = D^2
+ if $$\epsilon$$ = the euclidean distance, then D=0, the loss = 0
+ if $$\epsilon$$ < the euclidean distance, then D<0, the loss = 0

What this suggests is that the loss function only keeps the case where the euclidean distance is smaller than $$\epsilon$$ because there is no need to keep the two inputs further if their euclidean distance is less than or equal to the minimum distance $$\epsilon$$.

For the representation of all $$(x_i, x_j)$$ pairs' loss, we of course sum up all the pair's loss. 

### Noise Constrative Estimation (NCE)

Okay, now we move onto more modern approach for contrastive learning. [Noise contrastive estimation (NCE)](https://proceedings.mlr.press/v9/gutmann10a.html) distinguishes target from noise samples using logistic regression classification.

<img width="702" height="112" alt="image" src="https://github.com/user-attachments/assets/233e70f0-f432-48d5-a63f-e8f80f27ac53" />

Let x = target sample from the target distribution $$P_\theta (x) = P(x|C=1; \theta) $$

Also let $$\tilde{x}$$ = noise sample from the noise distribution
and 
$$Q (\tilde{x})=P(\tilde{x} | C=0)$$,
where C = label.

Logistic regression models log-odds. It models the logit of a sample u from the target distribution instead of the noise distribution:

$$l_\theta (u) = \log \frac{P_\theta (u)}{Q (u)} = \log P_\theta (u) - \log Q(u)$$

Then we can write the overall loss in the form of binary cross entropy:
$$-\frac{1}{N} \sum_{i=1}^N [\log \sigma (l_\theta (x_i)) + \log (1-\sigma (l_\theta (\tilde{x_i}))]$$

where $$\sigma(l)$$ is the sigmoid function:

$$\sigma(l_\theta (x)) = \frac{1}{1+\exp (-l_\theta (x)) } = \frac {1}{1+\frac{P_\theta (x)}{Q(x)}} = \frac{P_\theta (x)}{P_\theta (x) + Q (x)} = p(x \in T)$$ (T is the target set)

This works with only one target sample and a noise sample, but we can make this into considering multiple negative samples. 


### InfoNCE 

This is an advanced form of NCE, which uses categorical cross-entropy loss to capture a target sample from a set of unrelated noise samples.
We don't break the whole distribution down to two distributions for target and noise this time. 
The probability of finding the target sample correctly can be modeled as:

(c = context vector target sample from $$p(x|c)$$ while other N-1 samples are from p(x))

Label all samples as X = { $$x_i $$ } $$_{i=1}^N$$, where only one is target: $$x_t$$

$$p(C = 1 | X, c) = \frac {p(x_{t} | c) \Pi _{i=1, i \neq {t}}^N p(x_i)} {\sum_{j=1}^N [p(x_j|c) \Pi _ {i=1, i \neq {j}}^N p(x_i)]} = \frac{\frac{p(x_{t}|C)}{p(x_{t})}}{\sum_{j=1}^N \frac{ p(x_j |c)}{p(x_j)}} = \frac{f(x_{t}, c)}{\sum_{j=1}^N f(x_j, c)}$$




And the overall infoNCE loss is given by
<img width="326" height="74" alt="image" src="https://github.com/user-attachments/assets/d332b56c-e018-4278-b810-51102129138c" />

where  the scoring function $$f(x,c) \propto \frac{p(x|c)}{p(x)}$$. 
This thus means it optimizes negative log probability of classifying the target sample correctly. 

### Contrastive Language-Image Pre-training (CLIP)
<img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/cd607578-996b-4c88-8f7e-f0c43e2ed73c" />

Now let's get back to the approaches of specifically unifying multiple modalities. There has been an approach to map text and image modalities, called CLIP that was developed by OpenAI in 2021. The basic idea is building a one to one mapping between text and image. As the pipeline above suggests, for each pair of the text and image, both elements are encoded separately and then bridged by computing the cosine similarity (dot product) of the two, which creates one embedding space. This is a simple but a powerful tool. 

### ImageBind
<img width="1000" height="340" alt="image" src="https://github.com/user-attachments/assets/799058d6-f1b8-4936-86dd-019df5455616" />

Here's a SOTA approach for unifying multiple modalities called [ImageBind](https://arxiv.org/pdf/2305.05665), developed by Facebook in 2023.
The idea is to connect all modalities via the image modality. It binds modalities, including text, audio, video, depth, thermal, and IMU, by image data such that "all combinations of paired data are not necessary to train a joint embedding, and only image-paired data is sufficient to bind all modalities together".

ImageBind uses web-scale paired image and text data and uses naturally occurring pairs, such as video and audio and image and depth, etc. because that "allows for implicit aligning between text embeddings and other modalities without explicit semantic or textual pairing", which extends to zero shot recognition capabilities.

To do this, ImageBind aligns image embeddings to other modalities with contrastive learning.
Normally, like the case of CLIP, for zero-shot image classification with text prompts, the training of paired text data is required, but ImageBind does not because of this.

How the contrastive learning works in ImageBind is that it does self-supervised learning for pair (I, M) where I = image modality and M = another modality. Consider the pair ($$I_i$$, $$M_i$$). These elements of the pair are encoded into normalized embeddings $$q_i = f(I_i)$$ and $$k_i = g(M_i)$$ where f and g are DNNs. The embeddings and the encoders are then optimized using an InfoNCE loss:

<img width="370" height="58" alt="image" src="https://github.com/user-attachments/assets/676aaaef-22b2-44c3-b537-75b575668f52" />

where $$\tau$$ is a scalar temperature that changes smoothness of the softmax distribution, j = unrelated negatives.
Every example $$j \neq i$$ in the mini-batch is considered a negative. This makes the two normalized embeddings closer in the new unified embedding space, thereby aligning I and M. In actual practice, a symmetric loss of $$L_{I,M} + L_{M, I}$$ is employed.

The infoNCE loss looks a little different from the ordinary version. Well, it looks complex, yes, but if you look closely, you will notice that it just represents the probability of getting a positive pair in a softmax form with the  $$j \neq i$$ the negative pair, as I mentioned earlier. The goal should be to minimze the -log p(positive pair). 

This then enables zero-shot cross-modal retrieval tasks without explicit training. 

### Weaviate DB
<img width="578" height="378" alt="image" src="https://github.com/user-attachments/assets/13fc3322-e7e2-4f25-82e4-ab2fa08ee161" />

Using Weaviate's vector database, we can pull ImageBind via Docker and use it for practice. We can now perform a lot of interesting tasks. We can retrieve relevant audios and videos by giving a related natural language query, and we can also generate a image based on an audio input. 
Finally, we can also perform embedding-space arithmetic, where you can retrieve some content by adding up elements of different modalities. For instance, you can give an image of a pigeon and an audio of motor revving and retrieve an image with pigeons and a motorcycle. 

### MMRAG
<img width="566" height="484" alt="image" src="https://github.com/user-attachments/assets/df29f05b-6aa5-4201-bc2c-00692f130060" />

This indeed enables Multimodal RAG task by using vision LLMs (vLLMs). From the figure, the first step is about the retrieval task, where the prompt pulls a relevant image from Weaviate's vectorDB that contains data of all sorts of modalities in ImageBind. Then, the retrieved image and the main prompt is given to the LLM to create a custom response, which is the generation task. Here, we are retrieving specifically an image from the DB because of vision LLM's inability to receive audios or videos directly (as of 2024). Maybe in a near future, LLMs could also receive videos by zero shot, who knows?
