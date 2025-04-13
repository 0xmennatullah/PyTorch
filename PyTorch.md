- Funds?
	1. Traditional vs ML:
		1. trad: input + rules = output
			1. explain ability
			2. traditional approach better (simpler)
			3. unaccepting errors
			4. no enough data to extract rules
		2. ML || DL: input + output = rules
			1. good for the things that has million rules u can't throw outta ur mind
			2. environment change
			3. large data extracting insights (differentiate 100 kind of food, each has its own rules)
	2. ML vs DL:
		1. ML: 
			1. structured data: rows n columns
			2. production systems 
			3. ex: XG-Boost
		2. DL:
			4. un-structured data: posts, wikies, images, audio files
			5. ex: neural networks
			6. usages?
				1. ![[Pasted image ٢٠٢٥٠٤١٢٢٣٠٤١٥.png]]
		3. ![[Pasted image ٢٠٢٥٠٤١٢٢٢٢٦٤٩.png]]
	3. Neural Networks:
		1. input: unstructured data 
		2. turn it into numbers (numerical encoding)
		3. pass it to nn -> learns representation(weights)
		4. representation outputs(features, classes)
		5. outputs: human understandable form
		6. ![[Pasted image ٢٠٢٥٠٤١٢٢٢٥٤١٤.png]]
	4. Types of learning:
		1. supervised: data + labels foreach
		2. non-supervised: inherit representations | figure out differences wo labelling
		3. transfer: take patterns learned -> transfer it to another model t get ahead start
		4. reinforcement learning: environment + agent => rewards/no-reward
		5. ![[Pasted image ٢٠٢٥٠٤١٢٢٢٥٨٣٥.png]]
- PyTorch:
	1. GPU/TPU?
		1. cuda? 
			1. run ML code on GPU
			2. TPU: tensor processing unit -not as popular-
				1. tensor?
					1. ![[Pasted image ٢٠٢٥٠٤١٢٢٣١٥٥٢.png]]
					2. any numerical representation mainly.
					3. [Dan's](https://youtu.be/f5liqUk0ZTw?si=ERNNCMsPsRMAYRY6)
						1. vectors: can represent an area, make its length proportional to sqr meters of area + perpendicular on it
						2. v-components: projection of vector on the axis, so instead of drawing the vector, we can say it's n of X units, m of Y units, etc in Column vector!
						3. ![[Pasted image ٢٠٢٥٠٤١٢٢٣٢٥٤٧.png]]
						4. so, vectors are rank 1 tensors -> a basis vector for each direction, one directional component for each axis
						5. scalars -> tensors of rank zero, doesn't need directions
						6. forces + areas vectors in a matter for each surface => rank 2 tensor (2*2 matrix)
						7. 3-D matter representations -> (3*3 matrix) -> 3 indices for each component -> rank 3 tensor
						8. lillian liberes
	2. work flow: ![[Pasted image ٢٠٢٥٠٤١٢٢٣٣٤١٩.png]]
	3. intro to tensors:
		1. ==as1: read torch.tensor from doc==
		2. ![[Pasted image ٢٠٢٥٠٤١٣١٠٢٦٥٤.png]]
		3. anytime encoding data into numbers => torch.tensor
		4. we use num. brackets '[]' in tensors => depending of lvl of nesting, 2D => [], 3D => [[]], 4D=> [[[]]], etc.
		5. ![[Pasted image ٢٠٢٥٠٤١٣١٠٤٠٥٤.png]]
			1. tensors = batches of matrices.
			2. here, 1 matrix, 3 rows, 3 columns => (1,3,3) => tensor's shape.
			3. so outer [] => contains batches, inner [] contains 1st matrix/matrices in general, then [] contains each vector/element.
		6. 'tensor['batch'['matrix[vector[]]]]' => (1b, 1m,1v(r,c)).
		7. why MATRIX , TENSOR => CAP? no idea.
		8. ==as2: read random.tensors from doc==
		9. as3: torch.arange
		10. tensor_like => same shape of some tensor
			1. tensor1 = torch.zeros_like(tensor2)
			2. making a tensor - tensor1- with the shape of tensor2 but filled with zeros instead of elements, just got the structure only.
		11. data types:
			1. ==as4: datatype read from doc==
			2. precision0 in computing: num. of digits used to represent number
			3. single float point => 32 bit, 16 => half precision => less memo => faster cal
	