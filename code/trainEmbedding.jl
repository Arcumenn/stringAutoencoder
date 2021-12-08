using DelimitedFiles
using Flux
using Dates

const MAX_LENGTH = 25
const N_TRAINING = 2000
# const N_TRAINING = 200000 (value from the original code)
const hidden_size = 256
const SOS_token = 1
const EOS_token = 2
const teacher_forcing_ratio = 0.5


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


function trainIters(encoder, decoder, n_iters; print_every=1000, learning_rate=0.01)


    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every

    optimizer = Descent(learning_rate)
    
    ps = Flux.params(encoder, decoder)
    training_pairs = [vectorsFromPair(rand(pairs[1:N_TRAINING])) for i in 1:n_iters]
    
    p = Progress(Int(floor(n_iters / print_every)), showspeed=true)
    for iter in 1:n_iters
        training_pair = training_pairs[iter]
        input = training_pair[1]
        target = training_pair[2]

        loss = train!(ps, input, target, encoder, decoder, optimizer)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            next!(p; showvalues = [(:Iteration, iter), (:LossAverage, print_loss_avg)])
        end # if
    end # for
end # trainIters


function train!(ps, input, target, encoder, decoder, optimizer; max_length = MAX_LENGTH
    )::Float64

    input_length = length(input)
    target_length = length(target)

    loss::Float64 = 0

    for letter in input
        encoder(letter)
    end # for

    decoder_input::Int64 = SOS_token
    decoder[3].state = encoder[2].state

    use_teacher_forcing = rand() < teacher_forcing_ratio ? true : false
    # use_teacher_forcing = true

    gs = gradient(ps) do 
    if use_teacher_forcing
            # Teacher forcing: Feed the target as the next input
        for i in 1:length(target)   
            output = decoder(decoder_input)
                loss += Flux.Losses.mse(output, target[i])
                decoder_input = target[i]  # Teacher forcing
            end # for
            return loss
        else
            for i in 1:target_length
                output = decoder(decoder_input)
                println(length(output))
            topv, topi = findmax(output)
                println(topi)
                loss += Flux.Losses.mse(output, target[i])
                topi == EOS_token && break
        end # for
            return loss
    end # if/else
    end # do

    Flux.Optimise.update!(optimizer, ps, gs)
    final_loss = loss / target_length
    return final_loss
end

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

function ev_word(w, encoder, decoder)
    input_str = join(w, " ")
    output_str = evaluate(encoder, decoder, input_str)
    w_out = join(output_str[1:-1], "")
    distance = Levenshtein.distance(w)
    return distance / max(length(w), length(w_out))
end # ev_word


function evaluate(encoder, decoder, sentence; max_length=MAX_LENGTH)
    input = vectorFromSentence(input_lang, sentence)
    input_length = length(input)
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
end # evaluate

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
