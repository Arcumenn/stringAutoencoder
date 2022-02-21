using DelimitedFiles
using Flux
using ProgressMeter
using Statistics
using StringDistances
using Zygote

const MAX_LENGTH = 25
const N_TRAINING = 200000
const hidden_size = 256
const SOS_token = 1
const EOS_token = 2
const teacher_forcing_ratio = 0.5


mutable struct Language
    name::String
    word2index::Dict{String, Int32}
    word2count::Dict{String, Int32}
    index2word::Dict{Int32, String}
    n_words::Int32

    Language(name::String) = new(name, Dict{String, Int32}(), Dict{String, Int32}(),
                             Dict(1 => "SOS", 2 => "EOS"), 3)

end # Language


struct Autoencoder
    encoder::Chain
    decoder::Chain
    input_lang::Language
    output_lang::Language
    word_pairs::Vector{Vector{String}}
end # Autoencoder


function prepare_data(lang1, lang2, reverse::Bool, path::String
                     )::Tuple{Language, Language, Vector{Vector{String}}}

    input_lang, output_lang, lines = readlangs(lang1, lang2, reverse, path)
    println("Read $(Int(length(lines) / 2)) word pairs")

    pairs::Vector{Vector{String}} = [convert(Vector{String}, pair) for pair 
                                     in eachrow(lines) if filter_pair(pair)]

    println("Trimmed to $(length(pairs)) word pairs")
    println("Counting sounds...")
    for pair in pairs
        addsentence!(input_lang, pair[1])
        addsentence!(output_lang, pair[2])
    end # for

    println("Counted sounds:")
    println(input_lang.name, input_lang.n_words)
    println(output_lang.name, output_lang.n_words)

    # show the letters of the input language & their corresponding indeces
    word2index_dict = sort(collect(input_lang.word2index), by=x->x[2])
    show(word2index_dict)
    println()

    input_lang, output_lang, pairs
end # prepare_data


function readlangs(lang1::String, lang2::String, reverse::Bool=false, 
                   path_to_data::String=".")::Tuple{Language, Language, AbstractMatrix}

    println("Reading lines...")
    # Read the file and split into lines
    path::String = path_to_data * "/$(lang1)-$(lang2).txt"
    lines::AbstractMatrix = 
        DelimitedFiles.readdlm(path, '\t', AbstractString)

    # Reverse pairs, make Language instances
    if reverse
        @inbounds for r = 1:size(lines, 1)
            lines[r, 1], lines[r, 2] = lines[r, 2], lines[r, 1]
        end # @inbounds
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else
        input_lang = Language(lang1)
        output_lang = Language(lang2)
    end # if/else
    
    input_lang, output_lang, lines
end # readlangs


"""
    addsentence(lang::Language, sentence)

Function that iterates through a sentence to add all the words in the sentence to a lang
struct.
"""
function addsentence!(lang::Language, sentence::String)
    for word in split(sentence)
        addword!(lang, word)
    end # for
end # addsentence


"""
    addword!(lang::Language, word)

Function that adds a word to a Language struct & updates the fields of the language accordingly
"""
function addword!(lang::Language, word::SubString{String})
    if !(haskey(lang.word2index, word))
        lang.word2index[word] = lang.n_words
        lang.word2count[word] = 1
        lang.index2word[lang.n_words] = word
        lang.n_words += 1
    else
        lang.word2count[word] += 1
    end # if/else
end # addword!


"""
    filter_pair(p::SubArray{String})::Bool

This function checks if either of the 2 words in the pair exceeds the specified maximum
length. Returns true if both words are shorter then the maximum length; false otherwise. 
"""
function filter_pair(p::SubArray)::Bool
    return length(split(p[1])) < MAX_LENGTH && length(split(p[2])) < MAX_LENGTH
end # filter_pair


function pairs_to_vectors(pair::Vector{String}, languages::Tuple{Language, Language}
                         )::Tuple{Vector{Int32}, Vector{Int32}}

    input_vec = word_to_vector(languages[1], pair[1])
    target_vec = word_to_vector(languages[2], pair[2])
    return input_vec, target_vec
end # pairs_to_vectors


function word_to_vector(lang::Language, word::String)::Vector{Int32}
    indeces = [lang.word2index[letter] for letter in split(word)]
    append!(indeces, EOS_token)
    return indeces
end # word_to_vector


function train_iters!(encoder, decoder, word_pairs, langs::Tuple{Language, Language}, 
                      n_iters, alphabet_range::UnitRange, learning_rate::Float32, 
                      print_every::Int32)
    
    print_loss_total::Float32 = 0.0

    optimizer = Descent(learning_rate)

    loss::Float32 = 0.0

    ps = Flux.params(encoder, decoder)
    training_pairs = [pairs_to_vectors(rand(word_pairs[1:N_TRAINING]), langs) 
                      for i in 1:n_iters]
    
    p = Progress(Int(floor(n_iters / print_every)), showspeed=true)
    for iter in 1:n_iters
        training_pair = training_pairs[iter]
        input = training_pair[1]
        target = training_pair[2]
        
        loss, back = Flux.pullback(ps) do 
            model_loss(input, target, encoder, decoder, alphabet_range)  
        end # do
        
        grad = back(1f0)

        print_loss_total += loss / length(target)

        if iter % print_every == 0
            next!(p; showvalues = [(:Iteration, iter), 
                                   (:LossAverage, print_loss_total / print_every)])
            print_loss_total = 0
        end # if
        
        # update the model parameters with the optimizer based on the calculated gradients
        Flux.Optimise.update!(optimizer, ps, grad)

        # reset hidden state of encoder
        encoder[2].state = reshape(zeros(hidden_size), hidden_size, 1)
    end # for
end # train_iters


function model_loss(input::Vector{Int32}, target::Vector{Int32}, encoder, decoder, 
                    alphabet_range::UnitRange; max_length = MAX_LENGTH)::Float32

    word_length::Int32 = length(target)
    loss::Float32 = 0.0

    # let encoder encode the word
    for letter in input
        encoder(letter)
    end # for

    # set input and hidden state of decoder
    decoder_input::Int32 = SOS_token
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
    return loss
end # model_loss


function ev_word(w, m::Autoencoder)
    input_str = join(w, " ")
    output_str = evaluate(input_str, m)
    w_out = join(output_str[1:end-1], "")
    distance = Levenshtein()(w, w_out)
    return distance / max(length(w), length(w_out))
end # ev_word


function evaluate(word, m::Autoencoder; max_length=MAX_LENGTH)

    # reset hidden state of encoder
    encoder_hidden = reshape(zeros(hidden_size), hidden_size, 1)
    m.encoder[2].state = encoder_hidden

    input = word_to_vector(m.input_lang, word)

    # let encoder encode the word
    for letter in input
        m.encoder(letter)
    end # for

    # set first input and hidden state of decoder
    decoder_input::Int32 = SOS_token
    m.decoder[3].state = m.encoder[2].state

    decoded_words::Vector{String} = []

    for i in 1:max_length
        decoder_output = m.decoder(decoder_input)
        topv, topi = findmax(decoder_output)
        if topi == EOS_token
            push!(decoded_words, "<EOS>")
            break
        else
            push!(decoded_words, m.output_lang.index2word[topi])
        end # if/else
        decoder_input = topi
    end # for
    return decoded_words
end # evaluate


function get_embedding(encoder, lang, word)
    input = word_to_vector(lang, word)

    # reset hidden state of encoder
    encoder_hidden = reshape(zeros(hidden_size), hidden_size, 1)
    encoder[2].state = encoder_hidden

    for letter in input
        encoder(letter)
    end # for

    return encoder[2].state
end # get_embedding


"""
    train_model(;iters=75000, learning_rate=0.01, print_interval=500, device=cpu
               )::Autoencoder

Train an autoencoder. Set keyword arguments to use custom values for the number of
iterations, the learning rate, the print interval, the device (cpu/gpu) and the path to the
language data directory
"""
function train_model(;iters=75000, learning_rate=0.01, print_interval=500, device=cpu,
                     path::String="./data")::Autoencoder

    input_lang, output_lang, word_pairs = prepare_data("asjpIn", "asjpOut", false, path)

    encoderRNN = Chain(Flux.Embedding(input_lang.n_words - 1, hidden_size), 
                       GRU(hidden_size, hidden_size, init=Flux.kaiming_uniform)) |> device

    decoderRNN = Chain(Flux.Embedding(output_lang.n_words - 1, hidden_size), x -> relu.(x), 
                       GRU(hidden_size, hidden_size, init=Flux.kaiming_uniform), 
                       Dense(hidden_size, output_lang.n_words - 1, 
                       init=Flux.kaiming_uniform)
                      ) |> device

    train_iters!(encoderRNN, decoderRNN, word_pairs, (input_lang, output_lang), iters, 
                1:(input_lang.n_words - 1), Float32(learning_rate), Int32(print_interval))

    return Autoencoder(encoderRNN, decoderRNN, input_lang, output_lang, word_pairs)
end # train_model


"""
    test_model(m::Autoencoder; test_size::Int64=0)::Nothing

Test a Autoencoder model with word pairs that have not been used during training.
Prints the mean of the test results.
"""
function test_model(m::Autoencoder; test_size::Int64=0)::Nothing
    if test_size == 0 || (N_TRAINING + test_size  > length(m.word_pairs))
        testing = [join(split(x[1], " "), "") for x in m.word_pairs[N_TRAINING:end]]
    else
        testing = [join(split(x[1], " "), "") 
                   for x in m.word_pairs[N_TRAINING:(N_TRAINING + test_size)]]
    end # if/else
    testResults = @showprogress [ev_word(w, m) for w in testing]
    println("Mean Levenshtein distance between prediction and target letter: $(mean(testResults))")
end # test_model


"""
    save_embedding(m::Autoencoder; words=1000, path="../data/embedding.csv"
                  )::Nothing

Saves the embeddings of a specific number of words (default: 1000).
"""
function save_embedding(m::Autoencoder; words=1000, path="../data/embedding.csv"
                       )::Nothing

    embedding = [get_embedding(m.encoder, m.input_lang, p[1])[:,] 
                 for p in m.word_pairs[1:words]]
    writedlm(path, embedding, ",")
end # save_embedding