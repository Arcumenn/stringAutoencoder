using Dates
using DelimitedFiles
using Flux
using ProgressMeter
using Statistics
using StringDistances
using Zygote

const MAX_LENGTH = 25
const N_TRAINING = 200000
# const N_TRAINING = 200000 (value from the original code)
const hidden_size = 256
const SOS_token = 1
const EOS_token = 2
const teacher_forcing_ratio = 0.5


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
        DelimitedFiles.readdlm("../data/$(lang1)-$(lang2).txt", '\t', AbstractString)

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
                             Dict(1 => "SOS", 2 => "EOS"), 3)

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


function vectorsFromPair(pair::Vector{String})::Tuple{Vector{Int64}, Vector{Int64}}
    input_vec = vectorFromSentence(input_lang, pair[1])
    target_vec = vectorFromSentence(output_lang, pair[2])
    return input_vec, target_vec
end # vectorsFromPair


function vectorFromSentence(lang::Lang, sentence::String)::Vector{Int64}
    indeces = [lang.word2index[word] for word in split(sentence)]
    append!(indeces, EOS_token)
    return indeces
end # vectorFromSentence


function trainIters(encoder, decoder, n_iters; print_every=1000, learning_rate=0.01)
    
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every

    optimizer = Descent(learning_rate)
    
    local loss
    ps = Flux.params(encoder, decoder)
    # training_pairs = [vectorsFromPair(pairs[i]) for i in 1:n_iters]
    training_pairs = [vectorsFromPair(rand(word_pairs[1:N_TRAINING])) for i in 1:n_iters]
    
    p = Progress(Int(floor(n_iters / print_every)), showspeed=true)
    for iter in 1:n_iters
        training_pair = training_pairs[iter]
        input = training_pair[1]
        target = training_pair[2]
        
        loss, back = Flux.pullback(ps) do 
            model_loss(input, target, encoder, decoder)  
        end # do

        grad = back(1f0)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            next!(p; showvalues = [(:Iteration, iter), (:LossAverage, print_loss_avg)])
        end # if

        Flux.Optimise.update!(optimizer, ps, grad)
    end # for
end # trainIters


function model_loss(input, target, encoder, decoder; max_length = MAX_LENGTH)::Float64

    # reset hidden state of encoder
    encoder[2].state = reshape(zeros(hidden_size), hidden_size, 1)

    word_length::Int64 = length(target)
    alphabet_range::UnitRange = 1:(input_lang.n_words - 1)
    loss::Float64 = 0.0

    # let encoder encode the word
    for letter in input
        encoder(letter)
    end # for

    # set input and hidden state of decoder
    decoder_input::Int64 = SOS_token
    decoder[3].state = encoder[2].state

    use_teacher_forcing = rand() < teacher_forcing_ratio ? true : false

    if use_teacher_forcing
        # Teacher forcing: Feed the target as the next input
        for i in 1:word_length
            output = decoder(decoder_input)
            onehot_target = Flux.onehot(target[i], alphabet_range)
            loss += Flux.logitcrossentropy(output, onehot_target)
            decoder_input = target[i]
        end # for

    else
        for i in 1:word_length
            output = decoder(decoder_input)
            topv, topi = findmax(output)
            onehot_target = Flux.onehot(target[i], alphabet_range)
            loss += Flux.logitcrossentropy(output, onehot_target)
            decoder_input = topi
            if decoder_input == EOS_token 
                break
            end # if
        end # for
    end # if/else
    return loss / word_length
end # model_loss


function ev_word(w, encoder, decoder)
    input_str = join(w, " ")
    output_str = evaluate(encoder, decoder, input_str)
    w_out = join(output_str[1:end-1], "")
    distance = Levenshtein()(w, w_out)
    return distance / max(length(w), length(w_out))
end # ev_word


function evaluate(encoder, decoder, sentence; max_length=MAX_LENGTH)
    input = vectorFromSentence(input_lang, sentence)
    encoder_hidden = reshape(zeros(hidden_size), hidden_size, 1)
    encoder[2].state = encoder_hidden

    for letter in input
        encoder(letter)
    end # for

    decoder_input::Int64 = SOS_token
    decoder[3].state = encoder[2].state

    decoded_words::Vector{String} = []

    for i in 1:max_length
        decoder_output = decoder(decoder_input)
        topv, topi = findmax(decoder_output)
        decoder_input = topi
        if decoder_input == EOS_token
            push!(decoded_words, "<EOS>")
            break
        else
            push!(decoded_words, output_lang.index2word[topi])
        end # if/else
    end # for
    return decoded_words
end # evaluate



input_lang, output_lang, word_pairs = prepareData("asjpIn", "asjpOut", false)
word2index_dict = sort(collect(input_lang.word2index), by=x->x[2])
show(word2index_dict)

device = cpu

encoderRNN = Chain(Flux.Embedding(input_lang.n_words - 1, hidden_size), 
                   GRU(hidden_size, hidden_size)) |> device

decoderRNN = Chain(Flux.Embedding(output_lang.n_words, hidden_size), x -> relu.(x), 
                GRU(hidden_size, hidden_size), 
                Dense(hidden_size, output_lang.n_words - 1)) |> device

trainIters(encoderRNN, decoderRNN, 75000; print_every=5000)

testing = [join(split(x[1], " "), "") for x in word_pairs[N_TRAINING:end]]
# testing = [join(split(x[1], " "), "") for x in word_pairs[N_TRAINING:(N_TRAINING + 9999)]]

testResults = @showprogress [ev_word(w, encoderRNN, decoderRNN) for w in testing]

println((mean(testResults)))