using DelimitedFiles
using Flux
using Dates

const MAX_LENGTH = 25
const N_TRAINING = 2000
# const N_TRAINING = 200000 (value from the original code)
const hidden_size = 256

const SOS_token = 0
const EOS_token = 1


"""

"""
function prepareData(lang1, lang2, reverse::Bool=false
                    )::Tuple{Lang, Lang, Vector{Vector{String}}}

    input_lang, output_lang, lines = readLangs(lang1, lang2, reverse)
    println("Read $(Int(length(lines) / 2)) word pairs")

    pairs::Vector{Vector{String}} = [convert(Vector{String}, pair) for pair 
                                     in eachrow(lines) if filterPair(pair)]

    println("Trimmed to $(length(pairs)) word pairs")
    println("Counting sounds...")
    for pair in pairs
        addSentence!(input_lang, pair[1])
        addSentence!(output_lang, pair[2])
    end # for

    println("Counted sounds:")
    println(input_lang.name, input_lang.n_words)
    println(output_lang.name, output_lang.n_words)

    input_lang, output_lang, pairs
end


function readLangs(lang1::String, lang2::String, reverse::Bool=false
                  )::Tuple{Lang, Lang,AbstractMatrix}

    println("Reading lines...")
    # Read the file and split into lines
    lines::AbstractMatrix = 
        DelimitedFiles.readdlm("./data/$(lang1)-$(lang2).txt", '\t', AbstractString)

    # Reverse pairs, make Lang instances
    if reverse
        @inbounds for r = 1:size(lines, 1)
            lines[r, 1], lines[r, 2] = lines[r, 2], lines[r, 1]
        end # @inbounds
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    end # if/else
    
    input_lang, output_lang, lines
end


mutable struct Lang
    name::String
    word2index::Dict{String, Int64}
    word2count::Dict{String, Int64}
    index2word::Dict{Int64, String}
    n_words::Int64

    Lang(name::String) = new(name, Dict{String, Int64}(), Dict{String, Int64}(),
                             Dict(0 => "SOS", 1 => "EOS"), 2)

end


"""
    addSentence(lang::Lang, sentence)

Function that iterates through a sentence to add all the words in the sentence to a lang
struct.
"""
function addSentence!(lang::Lang, sentence::String)
    for word in split(sentence)
        addWord!(lang, word)
    end # for
end # addSentence


"""
    addWord!(lang::Lang, word)

Function that adds a word to a Lang struct & updates the fields of the language accordingly
"""
function addWord!(lang::Lang, word::SubString{String})
    if !(haskey(lang.word2index, word))
        lang.word2index[word] = lang.n_words
        lang.word2count[word] = 1
        lang.index2word[lang.n_words] = word
        lang.n_words += 1
    else
        lang.word2count[word] += 1
    end # if/else
end # addWord!


"""
    filterPair(p::SubArray{String})::Bool

This function checks if either of the 2 words in the pair exceeds the specified maximum
length. Returns true if both words are shorter then the maximum length; false otherwise. 
"""
function filterPair(p::SubArray)::Bool
    return length(split(p[1])) < MAX_LENGTH && length(split(p[2])) < MAX_LENGTH
end # filterPair


input_lang, output_lang, pairs = prepareData("asjpIn", "asjpOut", false)
word2index_dict = sort(collect(input_lang.word2index), by=x->x[2])
show(word2index_dict)


function vectorsFromPair(pair::Vector{String})::Tuple{Vector{Int64}, Vector{Int64}}
    input_vec = vectorsFromSentence(input_lang, pair[1])
    target_vec = vectorsFromSentence(output_lang, pair[2])
    return input_vec, target_vec
end # vectorsFromPair


function vectorsFromSentence(lang::Lang, sentence::String)::Vector{Int64}
    indeces = [lang.word2index[word] for word in split(sentence)]
    append!(indeces, EOS_token)
    return indeces
end # vectorsFromSentence


### Python Beginning ###
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
### Python End ###


 

encoderRNN = Chain(Flux.Embedding(input_lang.n_words, hidden_size), 
                   GRU(hidden_size, hidden_size))


function trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01)
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every

    # TO DO
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [vectorsFromPair(rand(pairs[1:N_TRAINING])) for i in 1:n_iters]
    # criterion = nn.NLLLoss()
    p = Progress(Int(floor(n_iters / print_every)), showspeed=true)
    for iter in 1:n_iters
        sleep(0.1)
        training_pair = training_pairs[iter]
        input_tensor = training_pair[1]
        target_tensor = training_pair[2]

        # loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            next!(p; showvalues = [(:Iteration, iter), (:LossAverage, print_loss_avg)])
        end # if
    end # for
end # trainIters

